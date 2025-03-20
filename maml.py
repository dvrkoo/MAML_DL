import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import higher
from torchvision import transforms
import random
from tqdm import tqdm

#################################
# Model Components (as in the paper)
#################################


def conv_block(
    in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True
):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    ]
    if use_bn:
        # Note: In the official implementation, batch norm momentum is set to 1.0 (for few examples)
        layers.append(nn.BatchNorm2d(out_channels, momentum=1.0, affine=True))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class MAMLConvNet(nn.Module):
    def __init__(self, n_way, in_channels=3, hidden_size=32):
        """
        Four-layer convolutional network as in the MAML paper.
        """
        super(MAMLConvNet, self).__init__()
        self.n_way = n_way

        self.layer1 = conv_block(in_channels, hidden_size)
        self.layer2 = conv_block(hidden_size, hidden_size)
        self.layer3 = conv_block(hidden_size, hidden_size)
        self.layer4 = conv_block(hidden_size, hidden_size)
        # Assuming input images are 84x84; after 4 poolings (factor 16 reduction) we get 84/16 ≈ 5.
        self.classifier = nn.Linear(hidden_size * 5 * 5, n_way)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits


#################################
# Meta Wrapper for MAML
#################################


class Meta(nn.Module):
    def __init__(self, args, model):
        """
        Meta wrapper that implements the inner-loop adaptation and meta-update.
        Args:
            args: Namespace or dict containing hyperparameters.
            model: The base learner (MAMLConvNet).
        """
        super(Meta, self).__init__()
        self.args = args
        self.net = model

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Performs the inner-loop adaptation and computes query loss.
        """
        # Create inner-loop optimizer for adaptation (ADAM)
        inner_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.inner_lr)
        # Use higher to create a functional copy of the network.
        with higher.innerloop_ctx(
            self.net, inner_optimizer, copy_initial_weights=False
        ) as (fnet, diffopt):
            for _ in range(self.args.update_step):
                logits = fnet(x_spt)
                loss = F.cross_entropy(logits, y_spt)
                diffopt.step(loss)
            # After adaptation, evaluate on the query set.
            logits_q = fnet(x_qry)
            qry_loss = F.cross_entropy(logits_q, y_qry)
            # Optionally, compute accuracy
            pred = torch.argmax(logits_q, dim=1)
            correct = torch.eq(pred, y_qry).sum().item()
            acc = correct / float(len(y_qry))
        return qry_loss, acc

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        Finetuning procedure for evaluation. This function returns the accuracy after each update step.
        It ensures that gradients are enabled for the inner-loop adaptation, even if called within a no_grad context.
        """
        inner_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.inner_lr)
        accs = []
        # Ensure gradients are enabled for inner-loop updates.
        with torch.set_grad_enabled(True):
            with higher.innerloop_ctx(
                self.net, inner_optimizer, track_higher_grads=False
            ) as (fnet, diffopt):
                # Evaluate before any updates.
                logits_q = fnet(x_qry)
                pred = torch.argmax(logits_q, dim=1)
                correct = torch.eq(pred, y_qry).sum().item()
                acc = correct / float(len(y_qry))
                accs.append(acc)
                # Finetuning updates.
                for _ in range(self.args.update_step_test):
                    logits = fnet(x_spt)
                    loss = F.cross_entropy(logits, y_spt)
                    diffopt.step(loss)
                    logits_q = fnet(x_qry)
                    pred = torch.argmax(logits_q, dim=1)
                    correct = torch.eq(pred, y_qry).sum().item()
                    acc = correct / float(len(y_qry))
                    accs.append(acc)
        return accs


#################################
# Training and Testing Functions
#################################


def maml_train(meta, meta_optimizer, data_loader, device="cuda"):
    meta.train()
    total_meta_loss = 0.0
    total_meta_acc = 0.0
    meta_batch_size = data_loader.batch_size

    # Process each episode in the meta batch separately.
    for support_images, support_labels, query_images, query_labels in tqdm(
        data_loader, desc="Training", leave=False
    ):
        # support_images: [meta_batch_size, n_way*k_shot, C, H, W]
        # query_images: [meta_batch_size, n_way*k_query, C, H, W]
        meta_loss = 0.0
        meta_acc = 0.0
        for i in range(meta_batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            qry_loss, acc = meta(x_spt, y_spt, x_qry, y_qry)
            meta_loss += qry_loss
            meta_acc += acc

        meta_loss = meta_loss / meta_batch_size
        meta_acc = meta_acc / meta_batch_size

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        total_meta_loss += meta_loss.item()
        total_meta_acc += meta_acc

    avg_meta_loss = total_meta_loss / len(data_loader)
    avg_meta_acc = total_meta_acc / len(data_loader)
    return avg_meta_loss, avg_meta_acc


def maml_test(meta, test_loader, device="cuda"):
    meta.eval()
    total_test_loss = 0.0
    total_test_acc = 0.0
    num_episodes = 0

    # with torch.no_grad():
    for support_images, support_labels, query_images, query_labels in test_loader:
        meta_batch_size = support_images.shape[0]
        for i in range(meta_batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            # Use finetunning to track performance over adaptation steps.
            accs = meta.finetunning(x_spt, y_spt, x_qry, y_qry)
            # We report the accuracy after full adaptation (final step).
            total_test_acc += accs[-1]
            num_episodes += 1

    avg_test_acc = total_test_acc / num_episodes
    return avg_test_acc


#################################
# Main Function to Reproduce MAML Paper Results
#################################


def main():
    # Hyperparameters modeled after the paper and official repo.
    class Args:
        n_way = 5
        k_shot = 1
        k_query = 15
        update_step = 5  # Inner loop steps for training.
        update_step_test = 10  # More inner loop steps for testing.
        inner_lr = 0.001
        meta_lr = 0.001
        episodes = 10000  # Total episodes for training.
        batch_size = 4  # Meta batch size (number of episodes per meta-update).
        epoch = 5  # Total training iterations (adjust as needed).

    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms for mini‑ImageNet.
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ]
    )

    # Create dataset using the CSV splits (make sure paths are updated)
    from datasets.dataloader import (
        MiniImageNetMetaDataset,
    )  # Import your CSV-based dataset.

    train_dataset = MiniImageNetMetaDataset(
        root="./datasets/MiniImageNet/",
        csv_file="./datasets/MiniImageNet/train.csv",
        num_classes=args.n_way,
        num_support=args.k_shot,
        num_query=args.k_query,
        transform=imagenet_transform,
        episodes=args.episodes,
    )
    test_dataset = MiniImageNetMetaDataset(
        root="./datasets/MiniImageNet/",
        csv_file="./datasets/MiniImageNet/test.csv",
        num_classes=args.n_way,
        num_support=args.k_shot,
        num_query=args.k_query,
        transform=imagenet_transform,
        episodes=100,  # Use fewer episodes for testing.
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    # Initialize the model and meta learner.
    model = MAMLConvNet(n_way=args.n_way).to(device)
    meta = Meta(args, model).to(device)
    meta_optimizer = optim.Adam(meta.parameters(), lr=args.meta_lr)

    global_step = 0
    for epoch in range(1, args.epoch + 1):
        # Create a tqdm progress bar for each epoch.
        pbar = tqdm(
            train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epoch}"
        )
        for support_images, support_labels, query_images, query_labels in pbar:
            meta_loss = 0.0
            meta_acc = 0.0
            meta_batch_size = support_images.shape[0]
            # Process each episode in the meta-batch separately.
            for i in range(meta_batch_size):
                x_spt = support_images[i].to(device)  # Shape: [n_way * k_shot, C, H, W]
                y_spt = support_labels[i].to(device)
                x_qry = query_images[i].to(device)  # Shape: [n_way * k_query, C, H, W]
                y_qry = query_labels[i].to(device)
                qry_loss, acc = meta(x_spt, y_spt, x_qry, y_qry)
                meta_loss += qry_loss
                meta_acc += acc
            meta_loss = meta_loss / meta_batch_size
            meta_acc = meta_acc / meta_batch_size

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            global_step += 1

            # Update progress bar every step with current loss.
            if global_step % 30 == 0:
                print("Train Loss:", meta_loss.item(), "Train Acc: ", meta_acc)
            # Evaluate on test episodes every 500 steps.
            if global_step % 500 == 0:
                test_acc = maml_test(meta, test_loader, device=device)
                pbar.write(f"[Step {global_step}] Test Acc: {test_acc:.2%}")
        # Optionally, print epoch summary.
        print(f"Epoch {epoch} completed.")


if __name__ == "__main__":
    main()
