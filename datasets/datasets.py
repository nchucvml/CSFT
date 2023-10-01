from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import csv
import os


class ADL(Dataset):
    def __init__(self, path, train, classes, transform, transform2, transform4):
        self.file_list = path
        self.train = train
        self.classes = classes
        self.transform = transform
        self.transform2 = transform2
        self.transform4 = transform4
        self.split_char = self.file_list[0][8]

    def __len__(self):
        if self.train:
            return len(self.file_list) * len(self.transform)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.file_list[idx // len(self.transform)]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform[idx % len(self.transform)])
            transformed2 = transforms.Compose(self.transform2[idx % len(self.transform)])
            transformed4 = transforms.Compose(self.transform4[idx % len(self.transform)])
        else:
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform)
            transformed2 = transforms.Compose(self.transform2)
            transformed4 = transforms.Compose(self.transform4)
        img_transformed = transformed(img)
        img_transformed2 = transformed2(img)
        img_transformed4 = transformed4(img)
        label = self.classes.index(img_path.split(self.split_char)[-2])    # normal=0 inflammation=1
        return img_transformed, img_transformed2, img_transformed4, label

    def each_class_num(self):
        each_class_num_list = [0 for i in range(len(self.classes))]
        for idx in range(len(self.file_list)):
            each_class_num_list[self.classes.index(self.file_list[idx].split(self.split_char)[-2])] += 1
        return each_class_num_list


class BreakHis(Dataset):
    def __init__(self, path, train, classes, transform, transform2, transform4):
        self.file_list = path
        self.train = train
        self.classes = classes
        self.transform = transform
        self.transform2 = transform2
        self.transform4 = transform4
        self.split_char = self.file_list[0][8]

    def __len__(self):
        if self.train:
            return len(self.file_list) * len(self.transform)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.file_list[idx // len(self.transform)]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform[idx % len(self.transform)])
            transformed2 = transforms.Compose(self.transform2[idx % len(self.transform)])
            transformed4 = transforms.Compose(self.transform4[idx % len(self.transform)])
        else:
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform)
            transformed2 = transforms.Compose(self.transform2)
            transformed4 = transforms.Compose(self.transform4)
        img_transformed = transformed(img)
        img_transformed2 = transformed2(img)
        img_transformed4 = transformed4(img)
        label = self.classes.index(img_path.split(self.split_char)[-1].split("_")[1])    # B=0 M=1
        information = [img_path.split(self.split_char)[-1].split("-")[3], img_path.split(self.split_char)[-1].split("-")[2]]    # [magnification, patient]
        return img_transformed, img_transformed2, img_transformed4, label, information

    def each_class_num(self):
        each_class_num_list = [0 for i in range(len(self.classes))]
        for idx in range(len(self.file_list)):
            each_class_num_list[self.classes.index(self.file_list[idx].split(self.split_char)[-1].split("_")[1])] += 1
        return each_class_num_list


class GlaS(Dataset):
    def __init__(self, path, train, classes, transform, transform2, transform4):
        self.train = train
        self.classes = classes
        self.transform = transform
        self.transform2 = transform2
        self.transform4 = transform4
        with open(os.path.join(path, "Grade.csv"), newline="") as csvfile:
            rows = csv.DictReader(csvfile)
            self.labels = {}
            for row in rows:
                self.labels[row["name"]] = row["grade (GlaS)"]    # key="name".value; value="grade (GlaS)".value
        if self.train:
            path = os.path.join(path, "train")
        else:
            path = os.path.join(path, "test")
        self.file_list = [os.path.join(path, r) for r in os.listdir(path)]
        self.split_char = self.file_list[0][8]

    def __len__(self):
        if self.train:
            return len(self.file_list) * len(self.transform)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.file_list[idx // len(self.transform)]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform[idx % len(self.transform)])
            transformed2 = transforms.Compose(self.transform2[idx % len(self.transform)])
            transformed4 = transforms.Compose(self.transform4[idx % len(self.transform)])
        else:
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform)
            transformed2 = transforms.Compose(self.transform2)
            transformed4 = transforms.Compose(self.transform4)
        img_transformed = transformed(img)
        img_transformed2 = transformed2(img)
        img_transformed4 = transformed4(img)
        label = self.classes.index(self.labels[img_path.split(self.split_char)[-1].split(".")[0]])    # benign=0 malignant=1
        return img_transformed, img_transformed2, img_transformed4, label

    def each_class_num(self):
        each_class_num_list = [0 for i in range(len(self.classes))]
        for idx in range(len(self.file_list)):
            each_class_num_list[self.classes.index(self.labels[self.file_list[idx].split(self.split_char)[-1].split(".")[0]])] += 1
        return each_class_num_list


class YTMF(Dataset):
    def __init__(self, path, train, classes, transform, transform2, transform4):
        self.file_list = path
        self.train = train
        self.classes = classes
        self.transform = transform
        self.transform2 = transform2
        self.transform4 = transform4
        self.split_char = self.file_list[0][8]

    def __len__(self):
        if self.train:
            return len(self.file_list) * len(self.transform)
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.train:
            img_path = self.file_list[idx // len(self.transform)]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform[idx % len(self.transform)])
            transformed2 = transforms.Compose(self.transform2[idx % len(self.transform)])
            transformed4 = transforms.Compose(self.transform4[idx % len(self.transform)])
        else:
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            transformed = transforms.Compose(self.transform)
            transformed2 = transforms.Compose(self.transform2)
            transformed4 = transforms.Compose(self.transform4)
        img_transformed = transformed(img)
        img_transformed2 = transformed2(img)
        img_transformed4 = transformed4(img)
        label = self.classes.index(img_path.split(self.split_char)[-1].split("_")[2][0])    # b=0 m=1
        return img_transformed, img_transformed2, img_transformed4, label

    def each_class_num(self):
        each_class_num_list = [0 for i in range(len(self.classes))]
        for idx in range(len(self.file_list)):
            each_class_num_list[self.classes.index(self.file_list[idx].split(self.split_char)[-1].split("_")[2][0])] += 1
        return each_class_num_list
