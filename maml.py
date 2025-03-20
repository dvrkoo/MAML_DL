import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import higher
from torchvision import transforms
import random
from tqdm import tqdm
from PIL import Image

# set random seed for reproducibility
random.seed(222)
#################################
# Model Components (as in the paper)
#################################


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MAMLConvNet(nn.Module):
    def __init__(self, n_way, in_channels=3, hidden_size=64):
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
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

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
        # Create inner-loop optimizer for adaptation (SGD)
        inner_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.inner_lr)
        # Use higher to create a functional copy of the network.
        with higher.innerloop_ctx(
            self.net, inner_optimizer, copy_initial_weights=True
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
        inner_optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.inner_lr)
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

    for batch_idx, (
        support_images,
        support_labels,
        query_images,
        query_labels,
    ) in enumerate(tqdm(data_loader)):
        meta_optimizer.zero_grad()
        task_accs = []  # Store accuracies for all tasks in this meta-batch
        losses = []

        for i in range(meta_batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            qry_loss, acc = meta(x_spt, y_spt, x_qry, y_qry)
            scaled_loss = qry_loss / meta_batch_size
            scaled_loss.backward(retain_graph=True)  # Backward on the scaled loss
            task_accs.append(acc)  # Save individual task accuracy
            losses.append(qry_loss.item())

        meta_optimizer.step()  # Update after all tasks
        torch.nn.utils.clip_grad_norm_(meta.parameters(), max_norm=0.5)

        # Print individual task accuracies for this meta-batch
        avg_acc = sum(task_accs) / len(task_accs)
        if batch_idx % 30 == 0:
            tqdm.write(f"\nMeta-Batch {batch_idx + 1}/{len(data_loader)}:")
            for i, acc in enumerate(task_accs):
                tqdm.write(f"  Task {i + 1}: Acc = {acc:.2%}, Loss = {losses[i]:.4f}")
            tqdm.write(f"  Avg Acc: {avg_acc:.2%}")

        # Track for epoch averages
        total_meta_loss += qry_loss.item() / meta_batch_size
        total_meta_acc += avg_acc

    avg_meta_loss = total_meta_loss / len(data_loader)
    avg_meta_acc = total_meta_acc / len(data_loader)
    return avg_meta_loss, avg_meta_acc


def maml_test(meta, test_loader, device="cuda"):
    meta.eval()
    step_accs = [
        [] for _ in range(meta.args.update_step_test + 1)
    ]  # Accuracies per step
    episode_count = 0

    for batch_idx, (
        support_images,
        support_labels,
        query_images,
        query_labels,
    ) in enumerate(test_loader):
        meta_batch_size = support_images.shape[0]
        for i in range(meta_batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            accs = meta.finetunning(x_spt, y_spt, x_qry, y_qry)

            # Print results for this episode
            tqdm.write(f"\nTest Episode {episode_count + 1}:")
            for step, acc in enumerate(accs):
                tqdm.write(f"  Step {step}: Acc = {acc:.2%}")
                step_accs[step].append(acc)

            episode_count += 1

    # Optionally, also return averaged results
    avg_step_accs = [sum(accs) / len(accs) for accs in step_accs]
    return avg_step_accs


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
        update_step_test = 5  # More inner loop steps for testing.
        inner_lr = 0.01
        meta_lr = 0.001
        episodes = 10000  # Total episodes for training.
        batch_size = 4  # Meta batch size (number of episodes per meta-update).
        epoch = 5  # Total training iterations (adjust as needed).

    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms for mini‑ImageNet.
    imagenet_transform = transforms.Compose(
        [
            # lambda x: Image.open(x).convert("RGB"),
            transforms.Resize((84, 84)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
    # Inside the main() training loop:
    global_step = 0
    for epoch in range(1, args.epoch + 1):
        avg_meta_loss, avg_meta_acc = maml_train(
            meta, meta_optimizer, train_loader, device
        )
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_meta_loss:.4f}")
        print(f"  Avg Acc: {avg_meta_acc:.2%}")

        # Evaluate every N epochs
        if epoch % 1 == 0:  # Adjust frequency as needed
            avg_step_accs = maml_test(meta, test_loader, device)
            print("\nTest Summary:")
            for step, acc in enumerate(avg_step_accs):
                print(f"  Step {step}: Avg Acc = {acc:.2%}")


if __name__ == "__main__":
    main()
