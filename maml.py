import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import higher
from torchvision import transforms
import random
from tqdm import tqdm
from datasets.dataloader import MiniImageNetMetaDataset

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
        # Change to Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

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
        super(Meta, self).__init__()
        self.args = args
        self.net = model
        # Match optimizer with reference implementation
        self.inner_optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.args.inner_lr
        )

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Performs the inner-loop adaptation and computes query loss.
        """
        with higher.innerloop_ctx(
            self.net, self.inner_optimizer, copy_initial_weights=True
        ) as (fnet, diffopt):
            # Inner loop adaptation
            for _ in range(self.args.update_step):
                logits = fnet(x_spt)
                loss = F.cross_entropy(logits, y_spt)
                diffopt.step(loss)

            # Evaluate on query set
            logits_q = fnet(x_qry)
            qry_loss = F.cross_entropy(logits_q, y_qry)

            # Calculate accuracy
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

    for batch_idx, (
        support_images,
        support_labels,
        query_images,
        query_labels,
    ) in enumerate(tqdm(data_loader)):
        meta_optimizer.zero_grad()
        batch_size = support_images.size(0)

        # Accumulate loss across tasks
        meta_batch_loss = 0
        task_accs = []

        # Process each task in the batch
        for i in range(batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            # Forward pass
            qry_loss, acc = meta(x_spt, y_spt, x_qry, y_qry)

            # Accumulate loss (normalize by batch size)
            meta_batch_loss += qry_loss / batch_size
            task_accs.append(acc)

        # Single backward pass on accumulated loss
        meta_batch_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(meta.parameters(), max_norm=0.5)

        # Update meta-parameters
        meta_optimizer.step()

        # Print progress
        avg_acc = sum(task_accs) / len(task_accs)
        if batch_idx % 30 == 0:
            tqdm.write(f"\nMeta-Batch {batch_idx + 1}/{len(data_loader)}:")
            for i, acc in enumerate(task_accs):
                tqdm.write(f"  Task {i + 1}: Acc = {acc:.2%}")
            tqdm.write(f"  Avg Acc: {avg_acc:.2%}, Loss: {meta_batch_loss.item():.4f}")

        # Track for epoch averages
        total_meta_loss += meta_batch_loss.item()
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
    # Hyperparameters matched to reference implementation
    class Args:
        n_way = 5
        k_shot = 1
        k_query = 15  # Standard for Mini-ImageNet
        update_step = 5  # Inner loop updates for training
        update_step_test = 10  # Inner loop updates for testing (more steps)
        inner_lr = 0.01  # Fast adaptation learning rate
        meta_lr = 0.001  # Meta learning rate
        batch_size = 4  # Meta batch size (tasks per update)
        episodes = 10000  # Total episodes to train
        epoch = 6  # For tracking progress (not actual epochs)

    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms for Mini-ImageNet
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Create datasets
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
        episodes=600,  # Fewer episodes for testing
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one task at a time for testing
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model with Xavier/Glorot initialization
    model = MAMLConvNet(n_way=args.n_way).to(device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    # Create meta-learner
    meta = Meta(args, model).to(device)
    meta_optimizer = optim.Adam(meta.parameters(), lr=args.meta_lr)

    # Training loop
    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        print(f"\n=== Epoch {epoch}/{args.epoch} ===")

        # Train
        avg_meta_loss, avg_meta_acc = maml_train(
            meta, meta_optimizer, train_loader, device
        )
        print(f"Train: Loss = {avg_meta_loss:.4f}, Accuracy = {avg_meta_acc:.2%}")

        # Test every 10 epochs
        if epoch % 1 == 0:
            avg_step_accs = maml_test(meta, test_loader, device)
            print("\nTest Results:")
            for step, acc in enumerate(avg_step_accs):
                print(f"  Step {step}: Avg Acc = {acc:.2%}")

            # Track best model
            final_acc = avg_step_accs[-1]
            if final_acc > best_acc:
                best_acc = final_acc
                torch.save(
                    meta.state_dict(),
                    f"maml_mini_imagenet_{args.n_way}way_{args.k_shot}shot_best.pt",
                )

        # Save checkpoint
        if epoch % 20 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": meta.state_dict(),
                    "optimizer_state_dict": meta_optimizer.state_dict(),
                    "loss": avg_meta_loss,
                },
                f"maml_mini_imagenet_{args.n_way}way_{args.k_shot}shot_epoch{epoch}.pt",
            )

    print(f"Training completed. Best accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
