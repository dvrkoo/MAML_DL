# MAML for Few-Shot Learning (PyTorch Implementation)

This repository contains a PyTorch implementation of Model-Agnostic Meta-Learning (MAML) specifically tailored for few-shot image classification tasks on the Omniglot and MiniImageNet datasets, as described in the paper "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" by Finn et al. (2017).

The implementation aims to replicate key aspects of the paper's methodology, including support for different network architectures (Convolutional and Fully Connected), first-order vs. second-order optimization, and dataset-specific hyperparameters. It integrates with [Comet ML](https://www.comet.com/) for experiment tracking and logging.

## Features

- Implementation of the MAML algorithm.
- Support for few-shot image classification.
- Datasets: Omniglot, MiniImageNet (using custom MetaDataset loaders).
- Models:
  - 4-layer Convolutional Network (`MAMLConvNet`) matching paper specifications (including BatchNorm).
  - 4-layer Fully Connected Network (`MAMLFCNet`) matching paper specifications (including BatchNorm and specified layer sizes).
- Optimization: First-order and Second-order MAML variants selectable via command-line flag.
- Logging: Integration with Comet ML for tracking metrics (loss, accuracy, gradients), hyperparameters, and system information.
- Evaluation: Includes testing loop (`maml_test`) and adaptation visualization script (`plot_adaptation.py`).
- Configurable Hyperparameters via `argparse`.

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- comet_ml
- numpy
- matplotlib (for plotting)
- tqdm (for progress bars)
- Pillow (PIL)

**Note on Backends:** Second-order MAML with the convolutional network requires higher-order gradients for `MaxPool2d`. While this works correctly on CUDA and CPU, the PyTorch MPS backend (Apple Silicon) requires `return_indices=True` to be set in `F.max_pool2d` calls within the model's functional forward pass to avoid runtime errors.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repo-url>
    cd <Repo-name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # .\.venv\Scripts\activate # Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    _Alternatively: `pip install torch torchvision torchaudio comet_ml numpy matplotlib tqdm Pillow` (adjust torch installation based on your system/CUDA version if needed)._

4.  **Comet ML Setup:**

    - Sign up for a free account at [Comet.com](https://www.comet.com/).
    - Get your API key from your Comet account settings.
    - Configure Comet ML:
      - **Recommended:** Set environment variables:
        ```bash
        export COMET_API_KEY="<Your-API-Key>"
        export COMET_PROJECT_NAME="maml_project" # Or your desired project name
        export COMET_WORKSPACE="<Your-Comet-Workspace>"
        ```
      - _Alternatively:_ Create a `.comet.config` file in the project root or your home directory (see Comet ML documentation).
      - _Or:_ Replace the API key directly in `maml.py` (not recommended for sharing).

5.  **Dataset Setup:**

    - **Omniglot:**
      - The dataset will be automatically downloaded and processed if it's not found in the expected location.
      - When creating the `OmniglotMetaDataset` in `maml.py`, ensure the `root` argument points to the desired base directory (e.g., `./datasets/omniglot`).
      - Set the `download=True` argument during the first run for either `background=True` (training) or `background=False` (evaluation) to trigger the download and extraction for that split. The data will be placed in a `processed` subdirectory within the `root`.
      - **Important:** The `OmniglotMetaDataset` class inherently includes 90-degree rotations by creating separate classes for each character rotation. No separate rotation transform is needed in the `transforms.Compose` pipeline for this dataset.
    - **MiniImageNet:**
      - Download the MiniImageNet dataset images and the standard CSV split files (`train.csv`, `val.csv`, `test.csv`). Common sources include [Ravi & Larochelle's splits](https://github.com/twitter-research/meta-learning-lstm) or other repositories hosting the data.
      - Place the extracted image folder (often named `images`) inside `./datasets/MiniImageNet/`.
      - Place the `train.csv`, `val.csv`, and `test.csv` files directly inside `./datasets/MiniImageNet/`. The structure should look like:
        ```
        ./datasets/MiniImageNet/
          ├── images/
          │   ├── n0153282900000005.jpg
          │   └── ... (all image files)
          ├── train.csv
          ├── val.csv
          └── test.csv
        ```
      - **Note:** Ensure you apply appropriate data augmentation (e.g., random flips, color jitter) via the `transform` argument for the _training_ `MiniImageNetMetaDataset` instance in `maml.py`.

6.  **Create Checkpoint Directory:**
    ```bash
    mkdir mp
    ```
    This directory (`mp`) will store the saved model checkpoints (`.pt` files).

## Usage

### Training

The main training script is `maml.py`. Use command-line arguments to configure the experiment based on the paper's settings or your own variations.

**Example Commands (Based on Paper Settings, ~30k Steps Target):**

- **Omniglot 5-way 1-shot FCNet:**

  ```bash
  python maml.py --omniglot \
      --inner_lr 0.4 --update_step 1 --batch_size 32 --update_step_test 3 \
      --epoch 16 # Check epoch calculation based on your dataset size/BS for target steps
  ```

- **MiniImageNet 5-way 5-shot ConvNet (Second Order):**

  ```bash
  python maml.py --conv --hidden_size 64 --k_shot 5 \
      --inner_lr 0.01 --update_step 5 --batch_size 2 --update_step_test 10 \
      --epoch 6
  ```

- **MiniImageNet 5-way 1-shot ConvNet (First Order):**

  ```bash
  python maml.py --conv --hidden_size 64 --first_order \
      --inner_lr 0.01 --update_step 5 --batch_size 4 --update_step_test 10 \
      --epoch 12
  ```

  ```
  _(See `maml.py --help` for all arguments and comments within the training commands for detailed hyperparameter justifications)_
  or just run trains.sh to reproduce paper results ;)
  ```

### Plotting Adaptation

Use the `plot_adaptation.py` script to visualize the fast adaptation performance on a single test task using a saved checkpoint.

```bash
python plot_adaptation.py <path_to_checkpoint.pt> \
    --n_way <N> \
    --k_shot <K> \
    --update_step_test <Steps> \
    --inner_lr <LR> \
    [--omniglot] [--conv] [--first_order] [--hidden_size <Size>] \
    [--baseline_checkpoint <path_to_baseline.pt>] [--baseline_lr <LR>]
```

## Project Structure

```
├── maml.py # Main training and evaluation script
├── plot_adaptation.py # Script to plot test-time adaptation
├── models.py # Contains MAMLConvNet, MAMLFCNet, Meta classes
├── datasets/
│ ├── dataloader.py # OmniglotMetaDataset, MiniImageNetMetaDataset classes
│ ├── omniglot/ # <--- Omniglot data automatically downloaded here (in ./processed subfolder)
│ └── MiniImageNet/ # <--- Place MiniImageNet data here
├── mp/ # <--- Saved model checkpoints (.pt files) appear here
├── requirements.txt # Python dependencies (Create this file)
└── README.md # This file

```

## Results

Training runs using the hyperparameters specified in the paper (adjusted for ~30k steps where applicable) yielded the following key observations:

- **Fast Adaptation:** MAML successfully demonstrates rapid adaptation on both Omniglot and MiniImageNet test tasks, often achieving high accuracy within 1-5 inner gradient steps, validating the core concept.
- **Architecture Impact:** Convolutional networks consistently outperformed Fully Connected Networks on both image datasets, highlighting the importance of spatial inductive biases.
- **Dataset Difficulty:** MiniImageNet proved significantly more challenging than Omniglot, resulting in lower overall accuracies, as expected due to image complexity.
- **First vs. Second Order:** On MiniImageNet, second-order MAML showed a slight performance advantage over the first-order approximation, albeit at a higher computational cost.
- **Augmentation & Hyperparameters:** Results indicate that data augmentation (rotations for Omniglot - handled internally by the dataset class; flips/jitter for MiniImageNet - applied via transforms) and careful tuning of hyperparameters (especially the sensitive `inner_lr=0.4` regime for Omniglot FCNet) are crucial for achieving benchmark performance.

## Acknowledgements

- I thank the authors of MAML (Chelsea Finn, Pieter Abbeel, Sergey Levine) for their foundational work.
- Gratitude to the creators and maintainers of the Omniglot and MiniImageNet datasets.
- This project utilizes the excellent [PyTorch](https://pytorch.org/) library and ecosystem.
- Experiment tracking and visualization were greatly aided by [Comet ML](https://www.comet.com/).
- _(Optional: Add any specific GitHub repositories or individuals whose code/ideas provided inspiration or assistance)._
