from comet_ml import Experiment
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from tqdm import tqdm
from datasets.dataloader import MiniImageNetMetaDataset, OmniglotMetaDataset
from models import MAMLFCNet, MAMLConvNet, Meta
import argparse


# Set random seed for reproducibility
random.seed(222)
torch.manual_seed(222)


experiment = Experiment(
    api_key="WQRfjlovs7RSjYUmjlMvNt3PY", project_name="maml_project", workspace="dvrkoo"
)

#################################
# Training and Testing Functions
#################################


def maml_test(meta, test_loader, device="cuda", epoch=None):
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
            # tqdm.write(
            #     f"\nTest Episode {episode_count+1}: "
            #     + ", ".join(
            #         [f"Step {step}: {acc:.2%}" for step, acc in enumerate(accs)]
            #     )
            # )
            for step, acc in enumerate(accs):
                step_accs[step].append(acc)
            episode_count += 1

    avg_step_accs = [sum(accs) / len(accs) for accs in step_accs]
    if experiment is not None:
        experiment.log_metrics(
            {
                f"test_step_{step}_accuracy": acc
                for step, acc in enumerate(avg_step_accs)
            },
            step=epoch,
        )
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
        # Log metrics more frequently with Comet
        if experiment is not None and batch_idx % 10 == 0:
            experiment.log_metrics(
                {
                    "batch_loss": meta_batch_loss.item(),
                    "batch_accuracy": avg_acc,
                    "gradient_norm": total_norm,
                },
                step=global_step,
            )
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
                    # f"./mp/maml_{"MiniImageNet" if }_step{global_step}_acc{final_acc:.4f}.pt",
                    f"./mp/maml{'_omniglot' if meta.args.omniglot else ''}_step{global_step}_k_{meta.args.k_shot}_n_way{meta.args.n_way}.pt",
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
    # class Args:
    #     n_way = 5
    #     k_shot = 1
    #     k_query = 15
    #     update_step = 5  # Inner-loop updates for training.
    #     update_step_test = 10  # Inner-loop updates for testing.
    #     inner_lr = 0.01  # Fast adaptation learning rate.
    #     meta_lr = 0.001  # Meta learning rate.
    #     batch_size = 4  # Meta batch size.
    #     episodes = 10000  # Total episodes for dataset.
    #     epoch = 5  # Number of training epochs.
    #     test_interval = 500  # Test every 500 steps
    #     first_order = False
    #     omniglot = True
    #     conv = False

    # let's make args with parseargs
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_query", type=int, default=15)
    parser.add_argument("--update_step", type=int, default=5)
    parser.add_argument("--update_step_test", type=int, default=10)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--test_interval", type=int, default=500)
    parser.add_argument("--first_order", action="store_true")
    parser.add_argument("--omniglot", action="store_true")
    parser.add_argument("--conv", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment.log_parameters(
        {
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "k_query": args.k_query,
            "inner_lr": args.inner_lr,
            "meta_lr": args.meta_lr,
            "first_order": args.first_order,
            "omniglot": args.omniglot,
            "update_step": args.update_step,
            "update_step_test": args.update_step_test,
        }
    )

    experiment.set_name(
        f"maml_{'omni' if args.omniglot else 'mini'}_{args.n_way}way_{args.k_shot}shot{'_conv' if args.conv else ''}_{'first' if args.first_order else 'second'}"
    )

    # Transforms for Mini-ImageNet.
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Example transform for Omniglot:
    omniglot_transform = transforms.Compose(
        [
            transforms.Resize(28),  # Resize to 28x28
            transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
            # transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
        ]
    )
    if not args.omniglot:
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
    else:
        train_dataset = OmniglotMetaDataset(
            root="./datasets/omniglot",
            num_classes=5,  # 5-way classification
            num_support=1,  # 1-shot learning
            num_query=15,
            transform=omniglot_transform,
            background=True,  # Use images_background
            episodes=60000,  # Match paper's 60k tasks
        )

        # Test set: 20 alphabets (evaluation) + rotations
        test_dataset = OmniglotMetaDataset(
            root="./datasets/omniglot",
            num_classes=5,
            num_support=1,
            num_query=15,
            transform=omniglot_transform,
            background=False,  # Use images_evaluation
            episodes=1000,  # Evaluate on 1k tasks
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
    if not args.omniglot:
        model = MAMLConvNet(n_way=args.n_way).to(device)
    else:
        model = MAMLConvNet(n_way=args.n_way, in_channels=1, hidden_size=64).to(device)

    if not args.conv:
        model = MAMLFCNet(n_way=args.n_way).to(device)

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
        # scheduler.step()
        # Log epoch metrics
        experiment.log_metrics(
            {
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=epoch,
        )

        print(f"Train: Loss = {avg_meta_loss:.4f}, Accuracy = {avg_meta_acc:.2%}")
        best_acc = max(best_acc, epoch_best_acc)

        # Run a full test at the end of each epoch
        print("\n=== Full Test at End of Epoch ===")
        avg_step_accs = maml_test(meta, test_loader, device, epoch)
        print("Test Results:")  # Apply first-order option if specified
        for step, acc in enumerate(avg_step_accs):
            print(f"  Step {step}: Avg Acc = {acc:.2%}")

        final_acc = avg_step_accs[-1]
        experiment.log_metrics(
            {
                "epoch_accuracy": final_acc,
            },
            step=epoch,
        )
        if final_acc > best_acc:
            best_acc = final_acc
            model_path = f"./mp/maml_{'omni' if args.omniglot else 'mini'}_{args.n_way}way_{args.k_shot}shot_epoch{epoch}{'_conv' if args.conv else ''}_{"first" if args.first_order else "second"}.pt"
            torch.save(meta.state_dict(), model_path)
            print(f"Current Best: {best_acc:.2%}")

    print(f"Training completed. Best accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
