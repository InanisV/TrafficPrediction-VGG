import csv
import torch
import torch.nn as tnn
import numpy
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from VGG16 import VGG16
from dataset import dataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

BATCH_SIZE = 64
LEARNING_RATE = 0.8
EPOCH = 200

file_path_v = 'dataset/V_228.csv'
time_slot = 2 * 12 - 1
predict_slot = 1

data_v = np.array(pd.read_csv(file_path_v, header=None).values)
part_len = len(data_v) // 10
train_data_v = data_v[: part_len * 9]
test_data_v = data_v[part_len * 9:]

train_set = dataset(train_data_v, time_slot, predict_slot, BATCH_SIZE)
test_set = dataset(test_data_v, time_slot, predict_slot, BATCH_SIZE)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

vgg16 = VGG16(num_classes=228)
vgg16.to(device)

# Loss and Optimizer
cost = tnn.MSELoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
# torch.save(vgg16, 'test.pt')

# evaluate model
def evaluate(loader):
    vgg16.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            data = torch.unsqueeze(data, 1)
            pred = vgg16(data)[1].detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.squeeze(np.hstack(predictions))
    labels = np.squeeze(np.hstack(labels))
    vgg16.train()
    return mean_squared_error(labels, predictions)

# train model
for epoch in range(EPOCH):
    avg_loss = 0
    for data, label in train_loader:
        data = torch.unsqueeze(data, 1)
        data, label = data.to(device), label.to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        features, output = vgg16(data)
        loss = cost(output, label)
        loss.backward()
        optimizer.step()
        avg_loss += loss
    avg_loss = avg_loss / len(train_loader)
    accuracy = evaluate(test_loader)
    lr = optimizer.param_groups[0]['lr']
    with open('logfile.txt', 'a') as f:
        f.write('Epoch: {:03d}, Loss: {:.5f}, Accuracy: {:.5f}, lr: {:.5f}\n'.format(epoch, avg_loss, accuracy, lr))
    with open('loss_record.csv', 'a') as f:
        csv_writer = csv.writer(f)
        record = [epoch, float(avg_loss), accuracy, lr]
        csv_writer.writerow(record)
    torch.save(vgg16, 'vgg16_test1.pt')
    # decrease learning rate
    scheduler.step()
