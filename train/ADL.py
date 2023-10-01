import os
import pandas
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from datasets.builder import loader
from models.CSFT import CrossScaleFusionTransformer


def train(model, record, train_loader, criterion, optimizer):
    model.train()
    total, batch_acc, batch_loss = 0, 0, 0.
    for batch_idx, (data1, data2, data3, labels) in enumerate(train_loader):
        data1, data2, data3, labels = data1.cuda(), data2.cuda(), data3.cuda(), labels.cuda()
        output = model(data1, data2, data3)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == labels).sum()
        total += len(labels)
        batch_acc += acc
        batch_loss += loss
    epoch_loss = batch_loss.item() / len(train_loader)
    epoch_acc = 100 * batch_acc.item() / total
    record["Lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
    record["Loss_train"].append(epoch_loss)
    record["Acc_train"].append(epoch_acc)
    print("Train_loss: {:.4f} Train_acc: {:.4f}%".format(epoch_loss, epoch_acc))


def test(model, record, test_loader, criterion, classes):
    class_correct = [0 for i in range(len(classes))]
    class_total = [0 for i in range(len(classes))]
    class_acc = []
    with torch.no_grad():
        model.eval()
        batch_loss = 0.
        for data1, data2, data3, labels in test_loader:
            data1, data2, data3, labels = data1.cuda(), data2.cuda(), data3.cuda(), labels.cuda()
            output = model(data1, data2, data3)
            loss = criterion(output, labels)
            _, predicted = torch.max(output, 1)
            pre = (predicted == labels).squeeze()
            batch_loss += loss
            for i in range(len(labels)):
                class_correct[labels[i]] += pre[i].item()
                class_total[labels[i]] += 1
        epoch_loss = batch_loss.item() / len(test_loader)
        epoch_acc = 100 * sum(class_correct) / sum(class_total)
        record["Loss_test"].append(epoch_loss)
        record["Acc_test"].append(epoch_acc)
        print("Test_loss: {:.4f} Test_acc: {:.4f}%".format(epoch_loss, epoch_acc))
        for i in range(len(classes)):
            class_acc.append(100 * class_correct[i] / class_total[i])
            print("Accuracy of %s: %f%%" % (classes[i], class_acc[i]))
        return epoch_acc, class_acc


def start(args, record, fold):
    train_loader, test_loader, classes, weight = loader(args, fold)
    model = CrossScaleFusionTransformer(args.img_size, len(classes))
    if args.train:
        model.load_from(np.load(args.pretrained_dir))
    else:
        if args.test_model is None:
            directory = os.path.join("weights", args.dataset + ".pt")
            print("--test_model is NONE, we use the default path.")
        else:
            directory = args.test_model
        print(directory)
        model.load_state_dict(torch.load(directory))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight).cuda()).to(device)
    best = 0.0
    epochs = args.epochs if args.train else 1
    for epoch in range(epochs):
        print("Fold" + str(fold) + " Epoch " + str(epoch+1) + "/" + str(epochs))
        s_time = time.perf_counter()
        if args.train:
            train(model, record, train_loader, criterion, optimizer)
        test_acc, class_acc = test(model, record, test_loader, criterion, classes)
        e_time = time.perf_counter()
        runtime = e_time - s_time
        print("Time: %d m %.3f s" % (runtime//60, runtime % 60))
        record["Epoch"].append(epoch + 1)
        record["Time"].append(runtime)
        record["Batch_size"].append(args.batch_size)
        if best <= test_acc:
            best = test_acc
            best_class_acc = class_acc
            if args.train:
                torch.save(model.state_dict(), os.path.join(args.save_dir, "fold" + str(fold) + ".pt"))
        record["Best"].append(best)
        record["Normal"].append(best_class_acc[0])
        record["Inflammation"].append(best_class_acc[1])
        print("Best: {:.4f}%".format(best))
        print("Best_normal: {:.4f}%".format(best_class_acc[0]))
        print("Best_inflammation: {:.4f}%".format(best_class_acc[1]))
        dataframe = pandas.DataFrame(record)
        csv_path = os.path.join(args.record_dir, "fold" + str(fold) + ".csv")
        dataframe.to_csv(csv_path, index=False, sep=",")


def initial(args, times=1):
    for fold in range(times):
        if args.train:
            record = {"Epoch": [], "Time": [], "Batch_size": [], "Lr": [],
                      "Loss_train": [], "Acc_train": [], "Loss_test": [], "Acc_test": [], "Best": [],
                      "Normal": [], "Inflammation": []}
        else:
            record = {"Epoch": [], "Time": [], "Batch_size": [],
                      "Loss_test": [], "Acc_test": [], "Best": [],
                      "Normal": [], "Inflammation": []}
        start(args, record, fold + 1)
