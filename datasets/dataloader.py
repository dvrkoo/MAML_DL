import os
import csv
import random
import urllib.request
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#################################
# Mini-ImageNet with CSV splits #
#################################


class MiniImageNetMetaDataset(Dataset):
    def __init__(
        self,
        root,
        csv_file,
        num_classes,
        num_support,
        num_query,
        transform=None,
        episodes=10000,
    ):
        """
        Args:
            root (str): Directory containing all images.
            csv_file (str): Path to the CSV file defining the split.
            num_classes (int): n-way.
            num_support (int): k-shot support examples per class.
            num_query (int): Number of query examples per class.
            transform: torchvision transforms.
            episodes (int): Number of episodes (dummy length, as episodes are generated on the fly).
        """
        self.root = root
        self.csv_file = csv_file
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.transform = transform
        self.episodes = episodes

        # Read CSV and group images by label.
        self.class_to_images = {}
        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                label = row["label"]
                path = os.path.join(self.root, filename)
                if label in self.class_to_images:
                    self.class_to_images[label].append(path)
                else:
                    self.class_to_images[label] = [path]

        # Only keep classes with enough images.
        self.classes = [
            cls
            for cls, imgs in self.class_to_images.items()
            if len(imgs) >= (self.num_support + self.num_query)
        ]
        if not self.classes:
            raise ValueError("No classes with enough images found in the CSV split.")

    def __len__(self):
        # Episodes are generated on the fly.
        return self.episodes

    def __getitem__(self, idx):
        # Randomly sample 'num_classes' classes.
        episode_classes = random.sample(self.classes, self.num_classes)
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for label, cls in enumerate(episode_classes):
            images = self.class_to_images[cls]
            samples = random.sample(images, self.num_support + self.num_query)
            support_samples = samples[: self.num_support]
            query_samples = samples[self.num_support :]

            for img_path in support_samples:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                support_images.append(image)
                support_labels.append(label)

            for img_path in query_samples:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                query_images.append(image)
                query_labels.append(label)

        # Convert lists to tensors.
        support_images = torch.stack(
            support_images
        )  # [num_classes*num_support, C, H, W]
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)  # [num_classes*num_query, C, H, W]
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels


#################################
# Omniglot with automatic download and processing #
#################################


class OmniglotMetaDataset(Dataset):
    urls = [
        "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
        "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip",
    ]

    def __init__(
        self,
        root,
        num_classes,
        num_support,
        num_query,
        transform=None,
        download=True,
        background=True,
        episodes=10000,
    ):
        """
        Args:
            root (str): Base directory where the raw and processed files will be stored.
            num_classes (int): n-way.
            num_support (int): k-shot support examples per class.
            num_query (int): Number of query examples per class.
            transform: torchvision transforms.
            download (bool): If True, downloads the dataset.
            background (bool): If True, uses the 'images_background' set; otherwise, 'images_evaluation'.
            episodes (int): Number of episodes.
        """
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.download = download
        self.background = background
        self.episodes = episodes

        if self.download:
            self._download_and_extract()

        # Define which subset to use.
        subset = "images_background" if self.background else "images_evaluation"
        self.data_path = os.path.join(self.root, "processed", subset)

        # Group images by character (i.e. class).
        self.class_to_images = {}
        for alphabet in os.listdir(self.data_path):
            alphabet_path = os.path.join(self.data_path, alphabet)
            if os.path.isdir(alphabet_path):
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    if os.path.isdir(character_path):
                        class_name = f"{alphabet}_{character}"
                        images = [
                            os.path.join(character_path, img)
                            for img in os.listdir(character_path)
                            if img.endswith(".png")
                        ]
                        if len(images) >= (self.num_support + self.num_query):
                            self.class_to_images[class_name] = images

        self.classes = list(self.class_to_images.keys())
        if not self.classes:
            raise ValueError("No classes with enough images found in Omniglot dataset.")

    def _download_and_extract(self):
        raw_folder = os.path.join(self.root, "raw")
        processed_folder = os.path.join(self.root, "processed")
        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)

        # Download each zip file if it does not exist.
        for url in self.urls:
            filename = url.split("/")[-1]
            filepath = os.path.join(raw_folder, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, filepath)
            # Extract contents to the processed folder.
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                print(f"Extracting {filename}...")
                zip_ref.extractall(processed_folder)

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        episode_classes = random.sample(self.classes, self.num_classes)
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for label, cls in enumerate(episode_classes):
            images = self.class_to_images[cls]
            samples = random.sample(images, self.num_support + self.num_query)
            support_samples = samples[: self.num_support]
            query_samples = samples[self.num_support :]

            for img_path in support_samples:
                # Omniglot images are grayscale.
                image = Image.open(img_path).convert("L")
                if self.transform:
                    image = self.transform(image)
                support_images.append(image)
                support_labels.append(label)

            for img_path in query_samples:
                image = Image.open(img_path).convert("L")
                if self.transform:
                    image = self.transform(image)
                query_images.append(image)
                query_labels.append(label)

        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels


#################################
# Example usage #
#################################

if __name__ == "__main__":
    # Define transforms for each dataset.
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
        ]
    )

    omniglot_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

    # Mini-ImageNet: update these paths to match your local structure.
    # Assume images are in 'mini_imagenet/images' and CSV splits are in 'mini_imagenet'.
    mini_root = "./MiniImageNet/"
    mini_csv_train = "./MiniImageNet/train.csv"
    mini_dataset = MiniImageNetMetaDataset(
        root=mini_root,
        csv_file=mini_csv_train,
        num_classes=5,
        num_support=1,
        num_query=15,
        transform=imagenet_transform,
        episodes=10000,
    )
    mini_loader = DataLoader(mini_dataset, batch_size=1, shuffle=True, num_workers=4)

    # Omniglot: specify a root folder where raw and processed data will reside.
    omniglot_root = "./omniglot"
    omniglot_dataset = OmniglotMetaDataset(
        root=omniglot_root,
        num_classes=5,
        num_support=1,
        num_query=5,
        transform=omniglot_transform,
        download=True,
        background=True,
        episodes=10000,
    )
    omniglot_loader = DataLoader(
        omniglot_dataset, batch_size=1, shuffle=True, num_workers=4
    )

    # Fetch one episode from mini-ImageNet.
    support_imgs, support_lbls, query_imgs, query_lbls = next(iter(mini_loader))
    print("Mini-ImageNet episode:")
    print(" Support images shape:", support_imgs.shape)
    print(" Query images shape:", query_imgs.shape)

    # Fetch one episode from Omniglot.
    support_imgs, support_lbls, query_imgs, query_lbls = next(iter(omniglot_loader))
    print("Omniglot episode:")
    print(" Support images shape:", support_imgs.shape)
    print(" Query images shape:", query_imgs.shape)
