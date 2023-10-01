from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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


def test(model, record, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        batch_loss = 0.
        for data1, data2, data3, labels in test_loader:
            data1, data2, data3, labels = data1.cuda(), data2.cuda(), data3.cuda(), labels.cuda()
            output = model(data1, data2, data3)
            loss = criterion(output, labels)
            output = torch.nn.functional.softmax(output, dim=1)
            probability, predicted = torch.max(output, 1)
            predicted = predicted.item()
            if predicted == 1:
                predicted = 1 if probability > 0.93 else 0
            pre = (predicted == labels).squeeze()
            batch_loss += loss
        epoch_loss = batch_loss.item() / len(test_loader)
        epoch_acc = 100 * int(pre.item())
        record["Loss_test"].append(epoch_loss)
        record["Acc_test"].append(epoch_acc)
        print("Test_loss: {:.4f} Test_acc: {:.4f}%".format(epoch_loss, epoch_acc))
        return predicted, labels.item()


def start(args, record, fold, error_file):
    train_loader, test_loader, classes, weight, file = loader(args, fold)
    model = CrossScaleFusionTransformer(args.img_size, len(classes))
    if args.train:
        model.load_from(np.load(args.pretrained_dir))
    else:
        if args.test_model is None:
            directory = os.path.join("weights", args.dataset, file + ".pt")
            print("--test_model is NONE, we use the default path.")
        else:
            directory = os.path.join(args.test_model, file + ".pt")
        print(directory)
        model.load_state_dict(torch.load(directory))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight).cuda()).to(device)
    predicted, label, best = -1, -1, 0.0
    epochs = args.epochs if args.train else 1
    for epoch in range(epochs):
        print("Fold" + str(fold) + " Epoch " + str(epoch+1) + "/" + str(epochs))
        s_time = time.perf_counter()
        if args.train:
            train(model, record, train_loader, criterion, optimizer)
        predicted, label = test(model, record, test_loader, criterion)
        e_time = time.perf_counter()
        runtime = e_time - s_time
        print("Time: %d m %.3f s" % (runtime//60, runtime % 60))
        print("Image %s: Pred %d Label %d" % (file, predicted, label))
        record["Epoch"].append(epoch+1)
        record["Time"].append(runtime)
        record["Batch_size"].append(args.batch_size)
        if args.train:
            torch.save(model.state_dict(), os.path.join(args.save_dir, file + ".pt"))
        dataframe = pandas.DataFrame(record)
        csv_path = os.path.join(args.record_dir, file + ".csv")
        dataframe.to_csv(csv_path, index=False, sep=",")
        if epoch+1 == epochs and predicted != label:
            error_file.append(file)
    return predicted, label


def initial(args, times=58):
    gt, predictions, error_file = [], [], []
    result = {"Precision": [], "Recall": [], "F1": [], "Accuracy": []}
    for fold in range(times):
        if args.train:
            record = {"Epoch": [], "Time": [], "Batch_size": [], "Lr": [],
                      "Loss_train": [], "Acc_train": [], "Loss_test": [], "Acc_test": []}
        else:
            record = {"Epoch": [], "Time": [], "Batch_size": [],
                      "Loss_test": [], "Acc_test": []}
        predicted, label = start(args, record, fold + 1, error_file)
        gt.append(label)
        predictions.append(predicted)
        file = open(os.path.join(args.record_dir, "error.txt"), "w")
        file.write(str(error_file))
        file.close()
    print(confusion_matrix(gt, predictions))
    result["Precision"].append(100 * precision_score(gt, predictions))
    result["Recall"].append(100 * recall_score(gt, predictions))
    result["F1"].append(100 * f1_score(gt, predictions))
    result["Accuracy"].append(100 * accuracy_score(gt, predictions))
    dataframe = pandas.DataFrame(result)
    csv_path = os.path.join(args.record_dir, "result.csv")
    dataframe.to_csv(csv_path, index=False, sep=",")
