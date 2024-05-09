import io

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class TrainDataset(Dataset):
    def __init__(self):
        # データセットの読み込み
        self.stl10_train = load_dataset("tanganke/stl10", split="train").to_pandas()
        self.cifar10_train = load_dataset("cifar10", split="train").to_pandas()
        self.cifar10_test = load_dataset("cifar10", split="test").to_pandas()
        self.cifarnet_train = load_dataset(
            "EleutherAI/cifarnet", split="train"
        ).to_pandas()
        self.cifarnet_test = load_dataset(
            "EleutherAI/cifarnet", split="test"
        ).to_pandas()
        self.cats_vs_dogs_train = load_dataset(
            "cats_vs_dogs", split="train", trust_remote_code=True
        ).to_pandas()
        self.monkey_species_collection_train = load_dataset(
            "Lehrig/Monkey-Species-Collection", split="train", trust_remote_code=True
        ).to_pandas()
        self.monkey_species_collection_test = load_dataset(
            "Lehrig/Monkey-Species-Collection", split="test", trust_remote_code=True
        ).to_pandas()
        self.oxford_iiit_pet_train = load_dataset(
            "timm/oxford-iiit-pet", split="train"
        ).to_pandas()
        self.oxford_iiit_pet_test = load_dataset(
            "timm/oxford-iiit-pet", split="test"
        ).to_pandas()
        self.horse2zebra_train = load_dataset(
            "gigant/horse2zebra", name="horse", split="train"
        ).to_pandas()
        self.horse2zebra_test = load_dataset(
            "gigant/horse2zebra", name="horse", split="test"
        ).to_pandas()
        self.imagenet_monkey_train = load_dataset(
            "yashikota/imagenet-monkey", split="train"
        ).to_pandas()

        # columnを変更
        self.cifar10_train.rename(columns={"img": "image"}, inplace=True)
        self.cifar10_test.rename(columns={"img": "image"}, inplace=True)
        self.cifarnet_train.rename(columns={"img": "image"}, inplace=True)
        self.cifarnet_test.rename(columns={"img": "image"}, inplace=True)
        self.cats_vs_dogs_train.rename(columns={"labels": "label"}, inplace=True)
        self.oxford_iiit_pet_train.drop(columns=["label", "image_id"], inplace=True)
        self.oxford_iiit_pet_test.drop(columns=["label", "image_id"], inplace=True)
        self.oxford_iiit_pet_train.rename(
            columns={"label_cat_dog": "label"}, inplace=True
        )
        self.oxford_iiit_pet_test.rename(
            columns={"label_cat_dog": "label"}, inplace=True
        )
        self.horse2zebra_train["label"] = 6
        self.horse2zebra_test["label"] = 6

        # pathをbytesに変換
        self.cats_vs_dogs_train = self._path_to_bytes(self.cats_vs_dogs_train)
        self.monkey_species_collection_train = self._path_to_bytes(
            self.monkey_species_collection_train
        )
        self.monkey_species_collection_test = self._path_to_bytes(
            self.monkey_species_collection_test
        )

        # labelの変換
        self.cifar10_train["label"] = self.cifar10_train["label"].apply(
            self._convert_cifar_label
        )
        self.cifar10_test["label"] = self.cifar10_test["label"].apply(
            self._convert_cifar_label
        )
        self.cifarnet_train["label"] = self.cifarnet_train["label"].apply(
            self._convert_cifar_label
        )
        self.cifarnet_test["label"] = self.cifarnet_test["label"].apply(
            self._convert_cifar_label
        )
        self.cats_vs_dogs_train["label"] = self.cats_vs_dogs_train["label"].apply(
            self._convert_cat_dog_label
        )
        self.oxford_iiit_pet_train["label"] = self.oxford_iiit_pet_train["label"].apply(
            self._convert_cat_dog_label
        )
        self.oxford_iiit_pet_test["label"] = self.oxford_iiit_pet_test["label"].apply(
            self._convert_cat_dog_label
        )
        self.monkey_species_collection_train["label"] = (
            self.monkey_species_collection_train[
                "label"
            ].apply(self._convert_monkey_label)
        )
        self.monkey_species_collection_test["label"] = (
            self.monkey_species_collection_test[
                "label"
            ].apply(self._convert_monkey_label)
        )
        self.imagenet_monkey_train["label"] = self.imagenet_monkey_train["label"].apply(
            self._convert_monkey_label
        )
        self.horse2zebra_train["label"] = self.horse2zebra_train["label"].apply(
            self._convert_horse2zebra_label
        )
        self.horse2zebra_test["label"] = self.horse2zebra_test["label"].apply(
            self._convert_horse2zebra_label
        )

        # labelがNoneの行を削除
        self.cifar10_train.dropna(subset=["label"], inplace=True)
        self.cifar10_test.dropna(subset=["label"], inplace=True)
        self.cifarnet_train.dropna(subset=["label"], inplace=True)
        self.cifarnet_test.dropna(subset=["label"], inplace=True)

        # columnのdtypeをintに変換
        self.cifar10_train["label"] = self.cifar10_train["label"].astype(int)
        self.cifar10_test["label"] = self.cifar10_test["label"].astype(int)
        self.cifarnet_train["label"] = self.cifarnet_train["label"].astype(int)
        self.cifarnet_test["label"] = self.cifarnet_test["label"].astype(int)

        self.combined_df = pd.concat(
            [
                self.cifar10_train,
                self.cifar10_test,
                self.cifarnet_train,
                self.cifarnet_test,
                self.stl10_train,
                self.cats_vs_dogs_train,
                self.monkey_species_collection_train,
                self.monkey_species_collection_test,
                self.oxford_iiit_pet_train,
                self.oxford_iiit_pet_test,
                self.horse2zebra_train,
                self.horse2zebra_test,
                self.imagenet_monkey_train,
            ]
        )

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize((96, 96)),
                v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
                # v2.AugMix(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _convert_cifar_label(self, label):
        if label == 6:  # frog
            return None
        elif label == 1:  # automobile -> car
            return 2
        elif label == 2:  # bird
            return 1
        elif label == 7:  # horse
            return 6
        else:
            return label

    def _convert_cat_dog_label(self, label):
        return 3 if label == 0 else 5

    def _convert_horse2zebra_label(self, _):
        return 6

    def _convert_monkey_label(self, _):
        return 7

    def _path_to_bytes(self, df):
        for i in range(len(df)):
            path = df.iloc[i]["image"]["path"]
            with open(path, "rb") as f:
                img_bytes = f.read()
            df.at[i, "image"]["bytes"] = img_bytes
            del df.at[i, "image"]["path"]

        return df

    def __len__(self):
        return len(self.combined_df)

    def __getitem__(self, idx):
        image = self.combined_df.iloc[idx]["image"]["bytes"]
        label = self.combined_df.iloc[idx]["label"]
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image = self.transform(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self):
        self.stl10 = load_dataset("tanganke/stl10", split="test").to_pandas()
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize((96, 96)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.stl10["label"] = self.stl10["label"].astype(int)

    def __len__(self):
        return len(self.stl10)

    def __getitem__(self, idx):
        image = self.stl10.iloc[idx]["image"]["bytes"]
        label = self.stl10.iloc[idx]["label"]
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image = self.transform(image)

        return image, label
