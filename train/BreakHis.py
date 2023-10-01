import csv
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
    for batch_idx, (data1, data2, data3, labels, _) in enumerate(train_loader):
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
    magnification = {"40": [[0, 0], [0, 0]], "100": [[0, 0], [0, 0]], "200": [[0, 0], [0, 0]], "400": [[0, 0], [0, 0]]}
    patient = {"40": {}, "100": {}, "200": {}, "400": {}}
    with torch.no_grad():
        model.eval()
        total, batch_acc, batch_loss = 0, 0, 0.
        for data1, data2, data3, labels, info in test_loader:
            data1, data2, data3, labels = data1.cuda(), data2.cuda(), data3.cuda(), labels.cuda()
            output = model(data1, data2, data3)
            loss = criterion(output, labels)
            _, predicted = torch.max(output, 1)
            pre = (predicted == labels).squeeze()
            acc = (output.argmax(dim=1) == labels).sum()
            total += len(labels)
            batch_acc += acc
            batch_loss += loss
            for i in range(len(labels)):
                magnification[info[0][i]][0][labels[i]] += pre[i].item()
                magnification[info[0][i]][1][labels[i]] += 1
                if not patient[info[0][i]].__contains__(info[1][i]):
                    patient[info[0][i]].setdefault(info[1][i], [[0, 0], [0, 0]])
                patient[info[0][i]][info[1][i]][0][labels[i]] += pre[i].item()
                patient[info[0][i]][info[1][i]][1][labels[i]] += 1
        epoch_loss = batch_loss.item() / len(test_loader)
        epoch_acc = 100 * batch_acc.item() / total
        record["Loss_test"].append(epoch_loss)
        record["Acc_test"].append(epoch_acc)
        print("Test_loss: {:.4f} Test_acc: {:.4f}%".format(epoch_loss, epoch_acc))
        print("Image Level")
        for m in magnification:
            print("  " + m + "X:")
            for i in range(len(classes)):
                print("    Accuracy of %s: %f%%" % (classes[i], 100 * magnification[m][0][i] / magnification[m][1][i]))
            magnification[m] = 100 * sum(magnification[m][0]) / sum(magnification[m][1])
            print("    Accuracy of %s: %f%%" % (m + "X", magnification[m]))
        print("Patient Level")
        for m in patient:
            patient_acc = []
            for p in patient[m]:
                patient_acc.append(100 * sum(patient[m][p][0])/sum(patient[m][p][1]))
            patient[m] = sum(patient_acc) / len(patient_acc)
            print("  Accuracy of %s: %f%%" % (m + "X", patient[m]))
        return epoch_acc, magnification, patient


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
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
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
        test_acc, magnification, patient = test(model, record, test_loader, criterion, classes)
        e_time = time.perf_counter()
        runtime = e_time - s_time
        print("Time: %d m %.3f s" % (runtime//60, runtime % 60))
        record["Epoch"].append(epoch+1)
        record["Time"].append(runtime)
        record["Batch_size"].append(args.batch_size)
        if best <= test_acc:
            best = test_acc
            if args.train:
                torch.save(model.state_dict(), os.path.join(args.save_dir, "fold" + str(fold) + ".pt"))
            result = []
            result.append(magnification)
            result.append(patient)
            header = ["40", "100", "200", "400"]
            with open(os.path.join(args.record_dir, "fold" + str(fold) + "_result.csv"), "w") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for elem in result:
                    writer.writerow(elem)
        record["Best"].append(best)
        print("Best: {:.4f}%".format(best))
        dataframe = pandas.DataFrame(record)
        csv_path = os.path.join(args.record_dir, "fold" + str(fold) + ".csv")
        dataframe.to_csv(csv_path, index=False, sep=",")


def initial(args, times=1):
    for fold in range(times):
        if args.train:
            record = {"Epoch": [], "Time": [], "Batch_size": [], "Lr": [],
                      "Loss_train": [], "Acc_train": [], "Loss_test": [], "Acc_test": [], "Best": []}
        else:
            record = {"Epoch": [], "Time": [], "Batch_size": [],
                      "Loss_test": [], "Acc_test": [], "Best": []}
        start(args, record, fold + 1)
