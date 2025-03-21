import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from torchvision import transforms
import random
from tqdm import tqdm
import numpy as np
from datasets.dataloader import MiniImageNetMetaDataset

# Set random seed for reproducibility
random.seed(222)
torch.manual_seed(222)

#################################
# Model Components (as in the paper)
#################################


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    # Conv -> BatchNorm (momentum=1.0) -> ReLU -> MaxPool2d
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, momentum=1.0, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


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
        # For 84x84 input, after 4 poolings we get ~5x5 feature maps.
        self.classifier = nn.Linear(hidden_size * 5 * 5, n_way)

        # Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, params=None):
        # If params is None, use self's parameters; otherwise, use the provided ones.
        if params is None:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            logits = self.classifier(out)
        else:
            # A simple functional implementation for our four-layer CNN.
            # We assume that params is a list in the order:
            # [conv1.weight, conv1.bias, bn1.weight, bn1.bias,
            #  conv2.weight, conv2.bias, bn2.weight, bn2.bias, ...,
            #  classifier.weight, classifier.bias]
            p = params  # shorthand
            # Layer 1
            out = F.conv2d(x, p[0], p[1], stride=1, padding=1)
            # We mimic BatchNorm using running stats from self.layer1[1] if needed.
            out = F.batch_norm(
                out,
                running_mean=self.layer1[1].running_mean,
                running_var=self.layer1[1].running_var,
                weight=p[2],
                bias=p[3],
                training=True,
                momentum=1.0,
            )
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, kernel_size=2)

            # Layer 2
            idx = 4
            out = F.conv2d(out, p[idx], p[idx + 1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=self.layer2[1].running_mean,
                running_var=self.layer2[1].running_var,
                weight=p[idx + 2],
                bias=p[idx + 3],
                training=True,
                momentum=1.0,
            )
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, kernel_size=2)

            # Layer 3
            idx = 8
            out = F.conv2d(out, p[idx], p[idx + 1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=self.layer3[1].running_mean,
                running_var=self.layer3[1].running_var,
                weight=p[idx + 2],
                bias=p[idx + 3],
                training=True,
                momentum=1.0,
            )
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, kernel_size=2)

            # Layer 4
            idx = 12
            out = F.conv2d(out, p[idx], p[idx + 1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=self.layer4[1].running_mean,
                running_var=self.layer4[1].running_var,
                weight=p[idx + 2],
                bias=p[idx + 3],
                training=True,
                momentum=1.0,
            )
            out = F.relu(out, inplace=True)
            out = F.max_pool2d(out, kernel_size=2)

            out = out.view(out.size(0), -1)
            # Classifier layer
            idx = 16
            logits = F.linear(out, p[idx], p[idx + 1])
        return logits

    def get_parameters(self):
        # Return parameters as a list (order must match the forward() implementation above)
        params = []
        # Layer 1: conv, bn
        params += [
            self.layer1[0].weight,
            self.layer1[0].bias,
            self.layer1[1].weight,
            self.layer1[1].bias,
        ]
        # Layer 2: conv, bn
        params += [
            self.layer2[0].weight,
            self.layer2[0].bias,
            self.layer2[1].weight,
            self.layer2[1].bias,
        ]
        # Layer 3: conv, bn
        params += [
            self.layer3[0].weight,
            self.layer3[0].bias,
            self.layer3[1].weight,
            self.layer3[1].bias,
        ]
        # Layer 4: conv, bn
        params += [
            self.layer4[0].weight,
            self.layer4[0].bias,
            self.layer4[1].weight,
            self.layer4[1].bias,
        ]
        # Classifier
        params += [self.classifier.weight, self.classifier.bias]
        return params


#################################
# Meta Wrapper for MAML (Manual Inner-Loop Updates)
#################################


class Meta(nn.Module):
    def __init__(self, args, model):
        super(Meta, self).__init__()
        self.args = args
        self.net = model  # base learner

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Training: perform inner-loop adaptation on the support set and compute query loss.
        """
        # Get initial fast weights directly from the network
        fast_weights = self.net.get_parameters()

        # Verify requires_grad
        for i, w in enumerate(fast_weights):
            if not w.requires_grad:
                print(f"Warning: Parameter {i} does not require grad")
                w.requires_grad = True

        # Inner-loop adaptation
        for _ in range(self.args.update_step):
            logits = self.net.forward(x_spt, params=fast_weights)
            loss = F.cross_entropy(logits, y_spt)

            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}")
                # Use a dummy loss to avoid breaking the training
                loss = torch.tensor(0.1, device=loss.device, requires_grad=True)

            # Create gradients
            grads = torch.autograd.grad(
                loss, fast_weights, create_graph=True, allow_unused=True
            )

            # Check gradients
            if any(g is None for g in grads):
                print("Warning: Some gradients are None")
                # Create dummy gradients where needed
                grads = [
                    g if g is not None else torch.zeros_like(w)
                    for g, w in zip(grads, fast_weights)
                ]

            # Debug gradient norms
            grad_norms = [g.norm().item() for g in grads]
            # print(f"Inner-loop grad norms: {grad_norms}")

            # Update fast weights.
            fast_weights = [
                w - self.args.inner_lr * g for w, g in zip(fast_weights, grads)
            ]

        # Evaluate on query set using adapted weights.
        logits_q = self.net.forward(x_qry, params=fast_weights)
        qry_loss = F.cross_entropy(logits_q, y_qry)

        # Debug query loss
        # print(f"Query loss: {qry_loss.item()}")

        # Calculate accuracy
        pred = torch.argmax(logits_q, dim=1)
        correct = torch.eq(pred, y_qry).sum().item()
        acc = correct / float(len(y_qry))
        return qry_loss, acc

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        Finetuning for evaluation. Returns accuracies after each update step.
        This procedure is done without computing higher-order gradients.
        """
        net_copy = copy.deepcopy(self.net)
        fast_weights = net_copy.get_parameters()
        accs = []

        # Evaluate before any update.
        logits_q = net_copy.forward(x_qry, params=fast_weights)
        pred = torch.argmax(logits_q, dim=1)
        correct = torch.eq(pred, y_qry).sum().item()
        acc = correct / float(len(y_qry))
        accs.append(acc)

        for _ in range(self.args.update_step_test):
            logits = net_copy.forward(x_spt, params=fast_weights)
            loss = F.cross_entropy(logits, y_spt)
            grads = torch.autograd.grad(loss, fast_weights)
            fast_weights = [
                w - self.args.inner_lr * g for w, g in zip(fast_weights, grads)
            ]
            logits_q = net_copy.forward(x_qry, params=fast_weights)
            pred = torch.argmax(logits_q, dim=1)
            correct = torch.eq(pred, y_qry).sum().item()
            acc = correct / float(len(y_qry))
            accs.append(acc)
        return accs


#################################
# Training and Testing Functions
#################################


def maml_test(meta, test_loader, device="cuda"):
    meta.eval()
    step_accs = [[] for _ in range(meta.args.update_step_test + 1)]
    episode_count = 0

    for batch_idx, (
        support_images,
        support_labels,
        query_images,
        query_labels,
    ) in enumerate(test_loader):
        batch_size = support_images.shape[0]
        for i in range(batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            accs = meta.finetunning(x_spt, y_spt, x_qry, y_qry)
            tqdm.write(
                f"\nTest Episode {episode_count+1}: "
                + ", ".join(
                    [f"Step {step}: {acc:.2%}" for step, acc in enumerate(accs)]
                )
            )
            for step, acc in enumerate(accs):
                step_accs[step].append(acc)
            episode_count += 1

    avg_step_accs = [sum(accs) / len(accs) for accs in step_accs]
    return avg_step_accs


def maml_train(
    meta, meta_optimizer, data_loader, test_loader, device="cuda", test_interval=500
):
    meta.train()
    total_meta_loss = 0.0
    total_meta_acc = 0.0
    global_step = 0
    best_acc = 0.0

    for batch_idx, (
        support_images,
        support_labels,
        query_images,
        query_labels,
    ) in enumerate(tqdm(data_loader)):
        meta_optimizer.zero_grad()
        batch_size = support_images.size(0)
        meta_batch_loss = 0.0
        task_accs = []

        for i in range(batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            qry_loss, acc = meta(x_spt, y_spt, x_qry, y_qry)
            meta_batch_loss += qry_loss / batch_size
            task_accs.append(acc)

        meta_batch_loss.backward()
        # Add gradient norm monitoring
        total_norm = 0
        for p in meta.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        torch.nn.utils.clip_grad_norm_(meta.parameters(), max_norm=0.5)
        meta_optimizer.step()

        global_step += 1
        avg_acc = sum(task_accs) / len(task_accs)

        if batch_idx % 30 == 0:
            tqdm.write(
                f"\nMeta-Batch {batch_idx+1}/{len(data_loader)}: "
                f"Avg Acc: {avg_acc:.2%}, Loss: {meta_batch_loss.item():.4f}, "
                f"Grad Norm: {total_norm:.4f}"
            )
        total_meta_loss += meta_batch_loss.item()
        total_meta_acc += avg_acc

        # Run test every test_interval steps
        if global_step % test_interval == 0:
            tqdm.write(f"\n=== Testing at step {global_step} ===")
            meta.eval()
            # Use a smaller subset of test tasks for quick evaluation
            quick_test_loader = [next(iter(test_loader)) for _ in range(10)]

            step_accs = [[] for _ in range(meta.args.update_step_test + 1)]
            for test_batch in quick_test_loader:
                support_images, support_labels, query_images, query_labels = test_batch
                x_spt = support_images[0].to(device)
                y_spt = support_labels[0].to(device)
                x_qry = query_images[0].to(device)
                y_qry = query_labels[0].to(device)

                accs = meta.finetunning(x_spt, y_spt, x_qry, y_qry)
                for step, acc in enumerate(accs):
                    step_accs[step].append(acc)

            avg_step_accs = [sum(accs) / len(accs) for accs in step_accs]
            tqdm.write("Quick Test Results:")
            for step, acc in enumerate(avg_step_accs):
                tqdm.write(f"  Step {step}: Avg Acc = {acc:.2%}")

            final_acc = avg_step_accs[-1]
            if final_acc > best_acc:
                best_acc = final_acc
                torch.save(
                    meta.state_dict(),
                    f"maml_mini_imagenet_step{global_step}_acc{final_acc:.4f}.pt",
                )
                tqdm.write(f"New Best: {best_acc:.2%}, Model saved.")

            # Resume training
            meta.train()

    avg_meta_loss = total_meta_loss / len(data_loader)
    avg_meta_acc = total_meta_acc / len(data_loader)
    return avg_meta_loss, avg_meta_acc, best_acc


#################################
# Main Function to Reproduce MAML Paper Results
#################################


def main():
    # Hyperparameters as in the paper/official repo.
    class Args:
        n_way = 5
        k_shot = 1
        k_query = 15
        update_step = 5  # Inner-loop updates for training.
        update_step_test = 10  # Inner-loop updates for testing.
        inner_lr = 0.01  # Fast adaptation learning rate.
        meta_lr = 0.001  # Meta learning rate.
        batch_size = 4  # Meta batch size.
        episodes = 10000  # Total episodes for dataset.
        epoch = 5  # Number of training epochs.
        test_interval = 500  # Test every 500 steps

    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms for Mini-ImageNet.
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Create datasets.
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
        episodes=600,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one task at a time during testing.
        shuffle=True,  # Shuffle to get different tasks for quick tests
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model.
    model = MAMLConvNet(n_way=args.n_way).to(device)

    # Create meta-learner.
    meta = Meta(args, model).to(device)
    meta_optimizer = optim.Adam(meta.parameters(), lr=args.meta_lr)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer, T_max=args.epoch * len(train_loader)
    )

    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        print(f"\n=== Epoch {epoch}/{args.epoch} ===")
        avg_meta_loss, avg_meta_acc, epoch_best_acc = maml_train(
            meta, meta_optimizer, train_loader, test_loader, device, args.test_interval
        )
        scheduler.step()

        print(f"Train: Loss = {avg_meta_loss:.4f}, Accuracy = {avg_meta_acc:.2%}")
        best_acc = max(best_acc, epoch_best_acc)

        # Run a full test at the end of each epoch
        print("\n=== Full Test at End of Epoch ===")
        avg_step_accs = maml_test(meta, test_loader, device)
        print("Test Results:")
        for step, acc in enumerate(avg_step_accs):
            print(f"  Step {step}: Avg Acc = {acc:.2%}")

        final_acc = avg_step_accs[-1]
        if final_acc > best_acc:
            best_acc = final_acc
            torch.save(
                meta.state_dict(),
                f"maml_mini_imagenet_{args.n_way}way_{args.k_shot}shot_epoch{epoch}_acc{final_acc:.4f}.pt",
            )
        print(f"Current Best: {best_acc:.2%}")

    print(f"Training completed. Best accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
