import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

Labels = ["ND"]


class BinaryClassificationDataset(Dataset):
    def __init__(self, imgPaths, transform=None, loader=default_loader):
        super(BinaryClassificationDataset, self).__init__()
        # self.annRoot = annRoot
        self.imgPaths = imgPaths
        self.transform = transform
        self.loader = loader
        self.LabelNames = Labels.copy()

        # gtPath = os.path.join(self.annRoot, "SewerML.csv")
        # gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=self.LabelNames + ["Filename"])
        # self.imgPaths = gt["Filename"].values
        # self.labels = gt[self.LabelNames].values

    def __len__(self):
        return (len(self.imgPaths))

    def __getitem__(self, index):
        path = self.imgPaths[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path


if __name__ == "__main__":
    from torch.utils.data import dataloader
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = BinaryClassificationDataset(annRoot="../annotations", imgRoot="../data", transform=transform)
    print(train_data.labels)
