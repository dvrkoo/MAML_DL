import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import os
import copy


# --- Assumed Imports ---
# Make sure these paths are correct relative to where you run the script
from models import MAMLFCNet, MAMLConvNet, Meta
from datasets.dataloader import MiniImageNetMetaDataset, OmniglotMetaDataset


torch.manual_seed(0)


# --- Helper Function to Recreate Args ---
# You might need to adjust this based on how you saved/know your args
def get_args_for_checkpoint(args_dict):
    parser = argparse.ArgumentParser()
    # Add all the arguments your training script uses,
    # especially those defining the model and task structure
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_query", type=int, default=15)  # Needed for dataset
    parser.add_argument(
        "--update_step", type=int, default=5
    )  # Used in Meta definition potentially
    parser.add_argument("--update_step_test", type=int, default=10)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument(
        "--meta_lr", type=float, default=0.001
    )  # Not needed for plotting
    parser.add_argument("--batch_size", type=int, default=1)  # Force 1 for plotting
    parser.add_argument(
        "--episodes", type=int, default=600
    )  # Not crucial for plotting one task
    parser.add_argument("--epoch", type=int, default=1)  # Not needed for plotting
    parser.add_argument("--test_interval", type=int, default=500)  # Not needed
    parser.add_argument("--first_order", action="store_true")  # IMPORTANT
    parser.add_argument("--omniglot", action="store_true")  # IMPORTANT
    parser.add_argument("--conv", action="store_true")  # IMPORTANT

    # Create a dummy namespace and update it from the dict
    # Provide default values that won't cause errors if missing from dict
    default_args = parser.parse_args([])
    default_args.__dict__.update(args_dict)  # Update with known values
    # Ensure boolean flags are set correctly if they exist in the dict
    if "first_order" in args_dict:
        default_args.first_order = args_dict["first_order"]
    if "omniglot" in args_dict:
        default_args.omniglot = args_dict["omniglot"]
    if "conv" in args_dict:
        default_args.conv = args_dict["conv"]

    return default_args


# --- Main Plotting Function ---
def plot_adaptation(
    maml_checkpoint_path,
    baseline_checkpoint_path=None,
    args_config=None,
    baseline_lr=0.01,
):
    """
    Loads models, performs adaptation on one test task, and plots accuracy curves.

    Args:
        maml_checkpoint_path (str): Path to the saved MAML .pt file.
        baseline_checkpoint_path (str, optional): Path to a saved baseline .pt file. Defaults to None.
        args_config (dict): Dictionary containing the hyperparameters used for the MAML model.
                            Example: {'n_way': 5, 'k_shot': 1, 'update_step_test': 10,
                                      'omniglot': False, 'conv': True, 'first_order': False,
                                      'inner_lr': 0.01}
        baseline_lr (float): Learning rate for fine-tuning the baseline model.
    """
    if args_config is None:
        # Define default or prompt user if no config provided
        # THIS IS CRUCIAL - MUST MATCH THE SAVED MODEL
        args_config = {
            "n_way": 5,
            "k_shot": 1,
            "update_step_test": 10,
            "omniglot": False,
            "conv": True,
            "first_order": False,
            "inner_lr": 0.01,
            "k_query": 15,  # Need k_query for dataset
        }
        print(
            "Warning: Using default args_config. Ensure this matches your saved MAML model!"
        )
        # You could add input() prompts here to ask the user

    args = get_args_for_checkpoint(args_config)
    args.batch_size = 1  # Override for single task plotting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(
        f"Plotting for N-Way={args.n_way}, K-Shot={args.k_shot}, Test Steps={args.update_step_test}"
    )
    print(
        f"Model: {'ConvNet' if args.conv else 'FCNet'}, Dataset: {'Omniglot' if args.omniglot else 'MiniImageNet'}"
    )
    print(
        f"MAML Order: {'First' if args.first_order else 'Second'}, Inner LR: {args.inner_lr}"
    )

    # --- 1. Load Data (Get ONE Test Task) ---
    if args.omniglot:
        omniglot_transform = transforms.Compose(
            [
                transforms.Resize(28),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        test_dataset = OmniglotMetaDataset(
            root="./datasets/omniglot",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=args.k_query,
            transform=omniglot_transform,
            background=False,
            episodes=10,  # Only need a few tasks
        )
    else:
        imagenet_transform = transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_dataset = MiniImageNetMetaDataset(
            root="./datasets/MiniImageNet/",
            csv_file="./datasets/MiniImageNet/test.csv",
            num_classes=args.n_way,
            num_support=args.k_shot,
            num_query=args.k_query,
            transform=imagenet_transform,
            episodes=10,  # Only need a few tasks
        )

    # Use shuffle=False to get the *same* task each time you run the script
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    try:
        (support_images, support_labels, query_images, query_labels) = next(
            iter(test_loader)
        )
    except StopIteration:
        print(
            "Error: Could not load data from test_loader. Is the dataset path correct?"
        )
        return

    # Extract the single task data (remove meta-batch dim)
    x_spt = support_images[0].to(device)
    y_spt = support_labels[0].to(device)
    x_qry = query_images[0].to(device)
    y_qry = query_labels[0].to(device)
    print(f"Loaded 1 task: Support shape={x_spt.shape}, Query shape={x_qry.shape}")

    # --- 2. Load MAML Model ---
    print(f"Loading MAML model from: {maml_checkpoint_path}")
    if args.conv:
        in_channels = 1 if args.omniglot else 3
        base_model_maml = MAMLConvNet(n_way=args.n_way, in_channels=in_channels).to(
            device
        )
    else:
        # Assuming FCNet input dim needs adjustment for Omniglot if used (e.g., 28*28=784)
        input_dim = (
            28 * 28 if args.omniglot else 84 * 84 * 3
        )  # Adjust if FCNet used differently
        base_model_maml = MAMLFCNet(n_way=args.n_way, input_dim=input_dim).to(device)

    maml_meta_model = Meta(args, base_model_maml).to(device)
    try:
        maml_checkpoint = torch.load(maml_checkpoint_path, map_location=device)
        # Handle potential saving variations (e.g., state_dict vs whole model)
        if isinstance(maml_checkpoint, dict) and "state_dict" in maml_checkpoint:
            maml_meta_model.load_state_dict(maml_checkpoint["state_dict"])
        elif isinstance(maml_checkpoint, dict) and not any(
            k.startswith("net.") for k in maml_checkpoint.keys()
        ):
            # Might be saved directly from Meta object
            maml_meta_model.load_state_dict(maml_checkpoint)
        elif isinstance(maml_checkpoint, dict) and any(
            k.startswith("net.") for k in maml_checkpoint.keys()
        ):
            # Saved state_dict likely already includes 'net.' prefix
            maml_meta_model.load_state_dict(maml_checkpoint)
        else:
            # Assuming it's the state_dict directly
            maml_meta_model.load_state_dict(maml_checkpoint)

        maml_meta_model.eval()
        print("MAML model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: MAML checkpoint file not found at {maml_checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading MAML state dict: {e}")
        print(
            "Ensure the args (conv, omniglot, n_way) match the saved model architecture."
        )
        return

    # --- 3. Perform MAML Adaptation ---
    print("Running MAML adaptation...")
    maml_accs_per_step = maml_meta_model.finetunning(x_spt, y_spt, x_qry, y_qry)
    print(f"MAML Accuracies per step: {[f'{acc:.2%}' for acc in maml_accs_per_step]}")

    # --- 4. Load Baseline Model and Fine-tune (Optional) ---
    baseline_model = None
    baseline_accs_per_step = []
    if baseline_checkpoint_path:
        print(f"Loading Baseline model from: {baseline_checkpoint_path}")
        try:
            # Instantiate baseline model (SAME ARCHITECTURE as MAML base)
            if args.conv:
                in_channels = 1 if args.omniglot else 3
                baseline_model = MAMLConvNet(
                    n_way=args.n_way, in_channels=in_channels
                ).to(
                    device
                )  # Or load from its own n_way if different
            else:
                input_dim = 28 * 28 if args.omniglot else 84 * 84 * 3
                baseline_model = MAMLFCNet(n_way=args.n_way, input_dim=input_dim).to(
                    device
                )

            baseline_checkpoint = torch.load(
                baseline_checkpoint_path, map_location=device
            )
            # Adjust loading based on how baseline was saved (might be state_dict directly)
            if (
                isinstance(baseline_checkpoint, dict)
                and "state_dict" in baseline_checkpoint
            ):
                baseline_model.load_state_dict(baseline_checkpoint["state_dict"])
            elif isinstance(baseline_checkpoint, dict):
                baseline_model.load_state_dict(baseline_checkpoint)
            else:  # Assume it's just the state_dict
                baseline_model.load_state_dict(baseline_checkpoint)
            baseline_model.eval()
            print("Baseline model loaded successfully.")

            # --- Perform Baseline Fine-tuning ---
            print(f"Running Baseline fine-tuning with LR={baseline_lr}...")
            baseline_model_copy = copy.deepcopy(baseline_model).to(device)
            baseline_model_copy.train()  # Set to train mode for updates

            # Use SGD or Adam - SGD often used for fine-tuning
            baseline_optimizer = torch.optim.SGD(
                baseline_model_copy.parameters(), lr=baseline_lr
            )

            # Evaluate at step 0
            baseline_model_copy.eval()
            with torch.no_grad():
                logits_q_base = baseline_model_copy(x_qry)
                pred_q_base = torch.argmax(logits_q_base, dim=1)
                correct_q_base = torch.eq(pred_q_base, y_qry).sum().item()
                acc_q_base = correct_q_base / float(len(y_qry))
                baseline_accs_per_step.append(acc_q_base)
            baseline_model_copy.train()

            # Fine-tuning loop
            for step in range(args.update_step_test):
                baseline_optimizer.zero_grad()
                logits_spt_base = baseline_model_copy(x_spt)
                loss_spt_base = F.cross_entropy(logits_spt_base, y_spt)
                loss_spt_base.backward()
                # Optional: Clip gradients for baseline? Usually not needed if LR is small
                # torch.nn.utils.clip_grad_norm_(baseline_model_copy.parameters(), max_norm=5)
                baseline_optimizer.step()

                # Evaluate on query set after each step
                baseline_model_copy.eval()
                with torch.no_grad():
                    logits_q_base = baseline_model_copy(x_qry)
                    pred_q_base = torch.argmax(logits_q_base, dim=1)
                    correct_q_base = torch.eq(pred_q_base, y_qry).sum().item()
                    acc_q_base = correct_q_base / float(len(y_qry))
                    baseline_accs_per_step.append(acc_q_base)
                baseline_model_copy.train()  # Set back for next update

            print(
                f"Baseline Accuracies per step: {[f'{acc:.2%}' for acc in baseline_accs_per_step]}"
            )

        except FileNotFoundError:
            print(
                f"Warning: Baseline checkpoint file not found at {baseline_checkpoint_path}. Skipping baseline."
            )
            baseline_model = None  # Ensure it's None if loading failed
        except Exception as e:
            print(f"Error loading or fine-tuning baseline model: {e}")
            baseline_model = None  # Ensure it's None if error occurred

    # --- 5. Plot the Results ---
    steps = list(range(args.update_step_test + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        maml_accs_per_step,
        marker="o",
        linestyle="-",
        linewidth=2,
        label=f"MAML ({args.n_way}w{args.k_shot}s)",
    )

    if baseline_model is not None and baseline_accs_per_step:
        plt.plot(
            steps,
            baseline_accs_per_step,
            marker="x",
            linestyle="--",
            label=f"Baseline Fine-tuning (LR={baseline_lr})",
        )

    # Extract MAML model filename for title
    maml_fname = os.path.basename(maml_checkpoint_path)
    plt.title(f"Few-Shot Adaptation on Single Task\nModel: {maml_fname}")
    plt.xlabel("Adaptation Steps (Inner Loop Updates)")
    plt.ylabel("Query Set Accuracy")
    plt.xticks(steps)
    plt.yticks([i / 10 for i in range(11)])  # Ticks every 10%
    plt.ylim(0.0, 1.05)  # Set y-axis limits
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_filename = f"adaptation_plot_{maml_fname.replace('.pt','')}.png"
    plt.savefig(plot_filename)
    print(f"\nSaved adaptation plot to: {plot_filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MAML adaptation performance on a single task."
    )
    parser.add_argument(
        "maml_checkpoint",
        help="Path to the trained MAML model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        default=None,
        help="Path to a trained baseline model checkpoint (.pt file) for comparison.",
    )
    parser.add_argument(
        "--baseline_lr",
        type=float,
        default=0.01,
        help="Learning rate for fine-tuning the baseline model.",
    )

    # --- Arguments needed to reconstruct the model and task ---
    # You MUST provide these values matching the saved MAML model
    parser.add_argument(
        "--n_way",
        type=int,
        required=True,
        help="N-way classification used during training.",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        required=True,
        help="K-shot learning used during training.",
    )
    parser.add_argument(
        "--update_step_test",
        type=int,
        required=True,
        help="Number of test updates used during training.",
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        required=True,
        help="Inner loop learning rate used during MAML training.",
    )
    parser.add_argument(
        "--k_query",
        type=int,
        default=15,
        help="K-query used for dataset loading (usually 15).",
    )  # Needed for dataset loading

    parser.add_argument(
        "--omniglot",
        action="store_true",
        help="Flag if the model was trained on Omniglot.",
    )
    parser.add_argument(
        "--conv",
        action="store_true",
        help="Flag if the model uses the convolutional architecture.",
    )
    parser.add_argument(
        "--first_order",
        action="store_true",
        help="Flag if the MAML model was trained with first-order approximation.",
    )
    # ---

    script_args = parser.parse_args()

    # Convert script args to the config dict needed by the function
    args_config_dict = {
        "n_way": script_args.n_way,
        "k_shot": script_args.k_shot,
        "k_query": script_args.k_query,
        "update_step_test": script_args.update_step_test,
        "inner_lr": script_args.inner_lr,
        "omniglot": script_args.omniglot,
        "conv": script_args.conv,
        "first_order": script_args.first_order,
        # Add any other args needed by your Meta or Model class init if necessary
    }

    plot_adaptation(
        maml_checkpoint_path=script_args.maml_checkpoint,
        baseline_checkpoint_path=script_args.baseline_checkpoint,
        args_config=args_config_dict,
        baseline_lr=script_args.baseline_lr,
    )
