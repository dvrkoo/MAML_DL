import os
import csv
import random
import urllib.request
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset

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
            episodes (int): Number of episodes.
        """
        self.root = root
        self.csv_file = csv_file
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.transform = transform
        self.episodes = episodes

        # Read CSV and group images by label
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

        # Only keep classes with enough images
        self.classes = [
            cls
            for cls, imgs in self.class_to_images.items()
            if len(imgs) >= (self.num_support + self.num_query)
        ]
        if not self.classes:
            raise ValueError("No classes with enough images found in the CSV split.")

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        # Randomly sample 'num_classes' classes
        episode_classes = random.sample(self.classes, self.num_classes)

        # Create support set
        support_images = torch.zeros(self.num_classes * self.num_support, 3, 84, 84)
        support_labels = torch.zeros(
            self.num_classes * self.num_support, dtype=torch.long
        )

        # Create query set
        query_images = torch.zeros(self.num_classes * self.num_query, 3, 84, 84)
        query_labels = torch.zeros(self.num_classes * self.num_query, dtype=torch.long)

        for i, cls in enumerate(episode_classes):
            # Get all images for this class
            images = self.class_to_images[cls]

            # Sample support and query images without replacement
            selected_imgs = random.sample(images, self.num_support + self.num_query)
            support_imgs = selected_imgs[: self.num_support]
            query_imgs = selected_imgs[
                self.num_support : self.num_support + self.num_query
            ]

            # Process support images
            for j, img_path in enumerate(support_imgs):
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                support_images[i * self.num_support + j] = img
                support_labels[i * self.num_support + j] = i  # Use index as label

            # Process query images
            for j, img_path in enumerate(query_imgs):
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_images[i * self.num_query + j] = img
                query_labels[i * self.num_query + j] = i  # Use index as label

        return support_images, support_labels, query_images, query_labels


#################################
# Omniglot with automatic download and processing #
#################################


class OmniglotMetaDataset(Dataset):
    urls = {
        "images_background": "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
        "images_evaluation": "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip",
    }

    def __init__(
        self,
        root,
        num_classes,
        num_support,
        num_query,
        transform=None,
        download=False,
        background=True,
        episodes=10000,
    ):
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.download = download
        self.background = background
        self.episodes = episodes
        self.class_to_images = {}

        if self.download:
            self._download_and_extract()

        # Define which subset to use
        subset = "images_background" if self.background else "images_evaluation"
        self.data_path = os.path.join(self.root, "processed", subset)

        if not os.path.exists(self.data_path):
            raise RuntimeError(f"Dataset not found. Use download=True")

        self._load_classes_with_rotations()

    def _load_classes_with_rotations(self):
        self.classes = []
        for alphabet in os.listdir(self.data_path):
            alphabet_path = os.path.join(self.data_path, alphabet)
            if not os.path.isdir(alphabet_path):
                continue
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                if not os.path.isdir(character_path):
                    continue
                # Add original + rotated versions
                for rotation in [0, 90, 180, 270]:
                    class_name = f"{alphabet}_{character}_rot{rotation}"
                    images = [
                        os.path.join(character_path, img)
                        for img in os.listdir(character_path)
                    ]
                    if len(images) >= (self.num_support + self.num_query):
                        self.class_to_images[class_name] = images
                        self.classes.append(class_name)

    def _download_and_extract(self):
        os.makedirs(self.root, exist_ok=True)
        subset = "images_background" if self.background else "images_evaluation"
        url = self.urls[subset]

        # Download
        zip_path = os.path.join(self.root, f"{subset}.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading {subset}...")
            urllib.request.urlretrieve(url, zip_path)

        # Extract
        print(f"Extracting {subset}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(self.root, "processed"))

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        episode_classes = random.sample(self.classes, self.num_classes)
        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for class_idx, class_name in enumerate(episode_classes):
            rotation = int(class_name.split("_rot")[-1])
            images = self.class_to_images[class_name]
            samples = random.sample(images, self.num_support + self.num_query)

            # Process support samples
            for path in samples[: self.num_support]:
                img = Image.open(path).convert("L").rotate(rotation)
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(class_idx)  # Assign label per sample

            # Process query samples
            for path in samples[self.num_support :]:
                img = Image.open(path).convert("L").rotate(rotation)
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(class_idx)  # Assign label per sample

        # Stack tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        return support_images, support_labels, query_images, query_labels
