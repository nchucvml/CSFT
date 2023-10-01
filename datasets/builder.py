from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
from datasets.datasets import ADL, BreakHis, GlaS, YTMF


def rw_txt(path, mode, content=None):
    if mode == "w":
        file = open(path, mode)
        file.write(str(content))
        file.close()
    else:
        txt = open(path, mode)
        txt_ = txt.read()
        txt_list = txt_.split(",")
        return txt_list


def multi_transforms(img_size, train):
    if train:
        trans = [[transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.RandomResizedCrop(img_size, scale=(1, 1), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.Resize((img_size, img_size)), transforms.RandomVerticalFlip(p=1), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.Resize((img_size, img_size)), transforms.RandomRotation(270), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.Resize((img_size, img_size)), transforms.RandomRotation(180), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                 [transforms.Resize((img_size, img_size)), transforms.RandomRotation(90), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]]
    else:
        trans = [transforms.Resize((img_size, img_size)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return trans


def loader(args, fold):
    train_transforms = multi_transforms(args.img_size, True)
    train_transforms2 = multi_transforms(args.img_size//2, True)
    train_transforms4 = multi_transforms(args.img_size//4, True)
    test_transforms = multi_transforms(args.img_size, False)
    test_transforms2 = multi_transforms(args.img_size//2, False)
    test_transforms4 = multi_transforms(args.img_size//4, False)
    directory = os.path.join(r"datasets", args.dataset)
    if args.dataset == "BreakHis":
        train_list = rw_txt(os.path.join("datasets", "BreakHis_train.txt"), "r")
        test_list = rw_txt(os.path.join("datasets", "BreakHis_test.txt"), "r")
        classes = ["B", "M"]
        train_data = BreakHis(train_list, True, classes, train_transforms, train_transforms2, train_transforms4)
        test_data = BreakHis(test_list, False, classes, test_transforms, test_transforms2, test_transforms4)
    elif args.dataset == "GlaS":
        classes = [" benign", " malignant"]
        train_data = GlaS(directory, True, classes, train_transforms, train_transforms2, train_transforms4)
        test_data = GlaS(directory, False, classes, test_transforms, test_transforms2, test_transforms4)
    elif args.dataset == "YTMF":
        train_list = [os.path.join(directory, r) for r in os.listdir(directory)]
        offsets = fold - 1
        test_list = [train_list[offsets]]
        del train_list[offsets]
        classes = ["b", "m"]
        train_data = YTMF(train_list, True, classes, train_transforms, train_transforms2, train_transforms4)
        test_data = YTMF(test_list, False, classes, test_transforms, test_transforms2, test_transforms4)
    else:
        directory = os.path.join(r"datasets", "ADL", args.dataset)
        train_list = rw_txt(os.path.join(directory, "train.txt"), "r")
        test_list = rw_txt(os.path.join(directory, "test.txt"), "r")
        classes = ["normal", "inflammation"]
        train_data = ADL(train_list, True, classes, train_transforms, train_transforms2, train_transforms4)
        test_data = ADL(test_list, False, classes, test_transforms, test_transforms2, test_transforms4)
    w = train_data.each_class_num()
    weight = [sum(w)/w[i] for i in range(len(classes))]
    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)
    if args.dataset == "YTMF":
        return train_loader, test_loader, classes, weight, test_list[0].split(directory[8])[-1]    # ../datasets/YTMF/***.tif
    else:
        return train_loader, test_loader, classes, weight
