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
import os


random.seed(222)
torch.manual_seed(222)


experiment = Experiment(api_key="yours", project_name="maml", workspace="dvrkoo")

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
            for step, acc in enumerate(accs):
                step_accs[step].append(acc)
            episode_count += 1

    avg_step_accs = [sum(accs) / len(accs) for accs in step_accs]
    return avg_step_accs


def maml_train(
    meta,  # The Meta model instance
    meta_optimizer,  # The optimizer for meta-parameters (e.g., Adam)
    data_loader,  # DataLoader for training tasks
    val_loader,  # DataLoader for validation tasks (used for periodic checks)
    device,  # Device ('cuda' or 'cpu')
    test_interval,  # How often (in global steps) to run quick validation
    global_step,  # Current global step count (passed in)
    epoch,  # Current epoch number (for logging/TQDM)
    scheduler=None,  # Optional learning rate scheduler
    max_grad_norm=0.5,  # Max norm for gradient clipping
):
    """
    Performs one epoch of MAML training.

    Args:
        meta: The Meta model instance.
        meta_optimizer: Optimizer for meta-parameters.
        data_loader: DataLoader for training tasks (yields batches of tasks).
        test_loader: DataLoader for validation tasks.
        device: Torch device.
        test_interval: Frequency (in global steps) for running quick validation.
        global_step: The current global step count before starting this epoch.
        epoch: The current epoch number.
        scheduler: Optional LR scheduler (stepped after each meta-update).
        max_grad_norm: Value for gradient clipping.

    Returns:
        avg_epoch_query_loss (float): Average query loss across all meta-batches in the epoch.
        avg_epoch_query_acc (float): Average query accuracy across all meta-batches in the epoch.
        best_val_acc_in_epoch (float): Best final-step validation accuracy found during periodic checks *within this epoch*.
        global_step (int): The updated global step count after finishing this epoch.
    """
    meta.train()

    epoch_total_query_loss = 0.0
    epoch_total_query_acc = 0.0
    best_val_acc_in_epoch = (
        0.0  # Tracks best validation accuracy found during this epoch's tests
    )

    # Iterate through meta-batches for one epoch
    for batch_idx, (
        support_images,
        support_labels,
        query_images,
        query_labels,
    ) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Training", leave=False)):

        meta_optimizer.zero_grad()
        batch_size = support_images.size(0)  # Number of tasks in this meta-batch

        meta_batch_query_loss_sum = 0.0  # Sum of query losses for tasks in the batch
        meta_batch_query_accs = []  # List of query accuracies for tasks in the batch
        meta_batch_inner_support_losses = (
            []
        )  # List of avg inner support losses per task
        meta_batch_inner_support_accs = []  # List of avg inner support accs per task

        # --- Inner Loop Simulation and Loss Accumulation ---
        for i in range(batch_size):
            x_spt = support_images[i].to(device)
            y_spt = support_labels[i].to(device)
            x_qry = query_images[i].to(device)
            y_qry = query_labels[i].to(device)

            qry_loss, qry_acc, task_avg_inner_loss, task_avg_inner_acc = meta(
                x_spt, y_spt, x_qry, y_qry
            )

            meta_batch_query_loss_sum += qry_loss  # Sum losses before averaging
            meta_batch_query_accs.append(qry_acc)
            meta_batch_inner_support_losses.append(task_avg_inner_loss)
            meta_batch_inner_support_accs.append(task_avg_inner_acc)

        # --- Meta-Update ---
        # 1. Calculate final average meta-loss for the batch
        final_meta_batch_query_loss = meta_batch_query_loss_sum / batch_size
        # 2. Compute gradients w.r.t. meta-parameters
        final_meta_batch_query_loss.backward()

        # 3. Calculate pre-clipping gradient norm (for logging)
        total_norm_sq = 0
        for p in meta.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        meta_gradient_norm_pre_clip = total_norm_sq**0.5

        # 4. Clip gradients
        torch.nn.utils.clip_grad_norm_(meta.parameters(), max_norm=max_grad_norm)

        # 5. Update meta-parameters
        meta_optimizer.step()
        if scheduler is not None:
            scheduler.step()  # Step scheduler after optimizer

        # 6. Increment global step counter
        global_step += 1

        # --- Calculate Averages for Logging ---
        avg_meta_batch_query_acc = (
            sum(meta_batch_query_accs) / len(meta_batch_query_accs)
            if meta_batch_query_accs
            else 0.0
        )
        avg_meta_batch_inner_loss = (
            sum(meta_batch_inner_support_losses) / len(meta_batch_inner_support_losses)
            if meta_batch_inner_support_losses
            else 0.0
        )
        avg_meta_batch_inner_acc = (
            sum(meta_batch_inner_support_accs) / len(meta_batch_inner_support_accs)
            if meta_batch_inner_support_accs
            else 0.0
        )

        # --- Logging to Comet (per meta-batch/step) ---
        if experiment is not None and batch_idx % 10 == 0:  # Log every 10 batches
            log_dict_train = {
                "train/meta_batch_query_loss": final_meta_batch_query_loss.item(),
                "train/meta_batch_query_accuracy": avg_meta_batch_query_acc,
                "train/meta_gradient_norm_pre_clip": meta_gradient_norm_pre_clip,
                "train/meta_batch_avg_inner_support_loss": avg_meta_batch_inner_loss,
                "train/meta_batch_avg_inner_support_accuracy": avg_meta_batch_inner_acc,
            }
            experiment.log_metrics(log_dict_train, step=global_step)

        if batch_idx % 30 == 0:
            tqdm.write(
                f"\nEpoch {epoch} | Batch {batch_idx+1}/{len(data_loader)} | Step {global_step} | "
                f"Query Acc: {avg_meta_batch_query_acc:.2%} | Query Loss: {final_meta_batch_query_loss.item():.4f} | "
                f"Grad Norm: {meta_gradient_norm_pre_clip:.4f}"
            )

        epoch_total_query_loss += final_meta_batch_query_loss.item()
        epoch_total_query_acc += avg_meta_batch_query_acc

        # --- Periodic Quick Validation ---
        if global_step % test_interval == 0:
            tqdm.write(f"\n=== Quick Validation at Step {global_step} ===")
            meta.eval()

            val_tasks_to_run = min(
                50, len(val_loader)
            )  # Number of tasks for quick validation
            quick_val_batches = []
            test_iter = iter(val_loader)  # Create iterator for test loader
            try:
                for _ in range(val_tasks_to_run):
                    quick_val_batches.append(next(test_iter))
            except StopIteration:
                tqdm.write(
                    f"Warning: Test loader exhausted after {len(quick_val_batches)} tasks for quick validation."
                )
                pass

            if not quick_val_batches:
                tqdm.write("Warning: No tasks available for quick validation.")
                meta.train()
                continue

            val_step_accs_all_tasks = [
                [] for _ in range(meta.args.update_step_test + 1)
            ]

            # Run finetuning on validation tasks
            # NO torch.no_grad() here, as finetuning needs internal grad calculation
            for val_batch in quick_val_batches:
                (
                    support_images_val,
                    support_labels_val,
                    query_images_val,
                    query_labels_val,
                ) = val_batch
                x_spt_val = support_images_val[0].to(device)
                y_spt_val = support_labels_val[0].to(device)
                x_qry_val = query_images_val[0].to(device)
                y_qry_val = query_labels_val[0].to(device)

                # finetunning returns list of accuracies per step
                accs_val_per_step = meta.finetunning(
                    x_spt_val, y_spt_val, x_qry_val, y_qry_val
                )

                for step_idx, acc in enumerate(accs_val_per_step):
                    if step_idx < len(val_step_accs_all_tasks):  # Safety check
                        val_step_accs_all_tasks[step_idx].append(acc)

            avg_val_step_accs = []
            for step_accs in val_step_accs_all_tasks:
                avg_val_step_accs.append(
                    sum(step_accs) / len(step_accs) if step_accs else 0.0
                )

            tqdm.write("Quick Validation Results:")
            for step, acc in enumerate(avg_val_step_accs):
                tqdm.write(f"  Step {step}: Avg Acc = {acc:.2%}")

            final_val_acc = avg_val_step_accs[-1]
            if final_val_acc > best_val_acc_in_epoch:
                best_val_acc_in_epoch = final_val_acc
                tqdm.write(
                    f"** New best quick validation accuracy during epoch: {best_val_acc_in_epoch:.2%} **"
                )

            meta.train()
        # --- End Periodic Validation ---

    # --- End of Epoch Calculation ---
    num_batches = len(data_loader)
    avg_epoch_query_loss = (
        epoch_total_query_loss / num_batches if num_batches > 0 else 0.0
    )
    avg_epoch_query_acc = (
        epoch_total_query_acc / num_batches if num_batches > 0 else 0.0
    )

    return avg_epoch_query_loss, avg_epoch_query_acc, best_val_acc_in_epoch, global_step


#################################
# Main Function to Reproduce MAML Paper Results
#################################


def main():
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
    parser.add_argument("--full_test_epoch_interval", type=int, default=1)
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
    imagenet_transform_train = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    imagenet_transform_test = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # transform for Omniglot:
    omniglot_transform_train = transforms.Compose(
        [
            # # Apply random 90-degree rotation BEFORE resizing/tensor conversion
            # transforms.Lambda(
            #     lambda img: transforms.functional.rotate(
            #         img, random.choice([0, 90, 180, 270]), fill=0
            #     )
            # ),  # fill=0 for black background
            transforms.Resize(28),  # Resize to 28x28
            transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
            # transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
        ]
    )
    omniglot_transform_test = transforms.Compose(
        [
            transforms.Resize(28),
            transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
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
            transform=imagenet_transform_train,
            episodes=args.episodes,
        )
        val_dataset = MiniImageNetMetaDataset(
            root="./datasets/MiniImageNet/",
            csv_file="./datasets/MiniImageNet/val.csv",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=args.k_query,
            transform=imagenet_transform_test,
            episodes=args.episodes,
        )

        test_dataset = MiniImageNetMetaDataset(
            root="./datasets/MiniImageNet/",
            csv_file="./datasets/MiniImageNet/test.csv",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=args.k_query,
            transform=imagenet_transform_test,
            episodes=600,
        )
    else:
        train_dataset = OmniglotMetaDataset(
            root="./datasets/omniglot",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=15,
            transform=omniglot_transform_train,
            background=True,
            episodes=60000,
        )
        val_dataset = OmniglotMetaDataset(
            root="./datasets/omniglot",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=15,
            transform=omniglot_transform_test,
            background=False,
            episodes=1000,
        )

        test_dataset = OmniglotMetaDataset(
            root="./datasets/omniglot",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=15,
            transform=omniglot_transform_test,
            background=False,
            episodes=1000,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
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
        model = MAMLConvNet(n_way=args.n_way, hidden_size=64).to(device)
    else:
        model = MAMLConvNet(n_way=args.n_way, in_channels=1, hidden_size=64).to(device)

    if not args.conv:
        model = MAMLFCNet(n_way=args.n_way).to(device)

    # Create meta-learner.
    meta = Meta(args, model).to(device)
    meta_optimizer = optim.Adam(meta.parameters(), lr=args.meta_lr)

    scheduler = None

    global_step = 0  # Initialize global step counter
    overall_best_val_acc = 0.0  # Track best validation accuracy across all epochs
    best_model_path = None  # Track best model path
    for epoch in range(1, args.epoch + 1):
        print(f"\n=== Epoch {epoch}/{args.epoch} Starting ===")

        avg_epoch_loss, avg_epoch_acc, best_val_acc_this_epoch, global_step = (
            maml_train(
                meta,
                meta_optimizer,
                train_loader,
                val_loader,
                device,
                args.test_interval,
                global_step,
                epoch,
                scheduler,
                max_grad_norm=0.5,
            )
        )

        if experiment is not None:
            experiment.log_metrics(
                {
                    "epoch/avg_query_loss": avg_epoch_loss,
                    "epoch/avg_query_accuracy": avg_epoch_acc,
                },
                epoch=epoch,
                step=global_step,
            )

        print(
            f"Epoch {epoch} Summary: Avg Query Loss = {avg_epoch_loss:.4f}, Avg Query Acc = {avg_epoch_acc:.2%}"
        )

        experiment.log_metrics(
            {
                "epoch/val_accuracy": best_val_acc_this_epoch,
            },
            step=global_step,
        )

        if best_val_acc_this_epoch > overall_best_val_acc:
            overall_best_val_acc = best_val_acc_this_epoch
            print(
                f"$$ New Overall Best Quick Validation Accuracy: {overall_best_val_acc:.2%} (found at step {global_step}) $$"
            )
            # Save the model based on this overall best validation accuracy
            model_path = f"./mp/maml_{'omni' if args.omniglot else 'mini'}_{args.n_way}way_{args.k_shot}shot_best_step{global_step}{'_conv' if args.conv else ''}_{'first' if args.first_order else 'second'}.pt"
            torch.save(meta.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            best_model_path = model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final testing...")

        if not args.omniglot:
            final_model = MAMLConvNet(n_way=args.n_way, hidden_size=64).to(device)
        else:
            final_model = MAMLConvNet(
                n_way=args.n_way, in_channels=1, hidden_size=64
            ).to(device)
        if not args.conv:
            final_model = MAMLFCNet(n_way=args.n_way).to(device)

        final_meta = Meta(args, final_model).to(device)
        final_meta.load_state_dict(torch.load(best_model_path))
        final_meta.eval()

        print("\n--- Running Final Test Set Evaluation ---")
        final_test_results = maml_test(
            final_meta, test_loader, device, epoch=args.epoch
        )

        print("\nFinal Test Results (using best validation model):")
        for step, acc in enumerate(final_test_results):
            print(f"  Step {step}: Avg Acc = {acc:.2%}")

        final_test_acc = final_test_results[-1]
        print(f"Final Test Accuracy: {final_test_acc:.2%}")
        if experiment is not None:
            experiment.log_metric("final/test_accuracy", final_test_acc)
            for step, acc in enumerate(final_test_results):
                experiment.log_metric(f"final/test_step_{step}_accuracy", acc)

    else:
        print(
            "No best model was saved during training (or path not found). Cannot run final test."
        )

    experiment.end()

    print(
        f"\nTraining completed. Best quick validation accuracy achieved: {overall_best_val_acc:.2%}"
    )


if __name__ == "__main__":
    main()
