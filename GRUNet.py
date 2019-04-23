import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_pd = pd.read_csv('train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_size = 1
batch_size = 64
seq_length = 128
hidden_size = 64
step_length = 32


class GRUNet(nn.Module):

    def __init__(self, batch_size, feature_size, hidden_size, seq_len):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(
            input_size = feature_size,
            hidden_size = hidden_size,
            batch_first = True,
            bidirectional = False,
        )
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        # self.fc1 = nn.Linear(hidden_size, 16 * seq_len)
        # self.fc2 = nn.Linear(16 * seq_len, seq_len)

    def forward(self, x):
        x = x.contiguous().view(self.batch_size, self.seq_len, self.feature_size)
        x, _ = self.gru(x)
        x = x[:, :, -1]
        x = x.contiguous().view(self.batch_size, self.seq_len * self.feature_size)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x


class LANLDataset(Dataset):

    # Apply sliding window to the dataset, each input window size is seq_len and stride with a step length
    def __init__(self, pd, seq_len, step_length):
        super(LANLDataset, self).__init__()
        self.pd = pd
        self.X = pd.acoustic_data.values
        self.y = pd.time_to_failure.values
        self.seq_len = seq_len
        self.step_length = step_length

    def __len__(self):
        return int((len(self.pd) - self.seq_len + 1) / self.step_length)

    def __getitem__(self, index):
        return self.X[index * self.step_length : index * self.step_length + self.seq_len], \
            self.y[index * self.step_length : index * self.step_length + self.seq_len]


def train():
    net = GRUNet(batch_size, feature_size, hidden_size, seq_length)
    dataset = LANLDataset(train_pd, seq_length, step_length)
    val_size = 50085877
    dataset_val = Subset(dataset, range(val_size))
    dataset_train = Subset(dataset, range(val_size, len(dataset)))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = batch_size, shuffle = False, 
        pin_memory = True, num_workers = 8, drop_last = True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size = batch_size, shuffle = False, 
        pin_memory = True, num_workers = 8, drop_last = True
    )

    loss_func = nn.L1Loss()
    optimi = optim.Adam(net.parameters(), lr = 0.01)

    net.to(device)

    for epoch in range(200):
        net.train()
        train_losses = []
        val_losses = []

        step = 1

        for inputs, outputs in tqdm(iter(train_loader)):
            inputs = inputs.to(device).float()
            outputs = outputs.to(device).float()
            preds = net(inputs)
            loss = loss_func(preds, outputs)
            optimi.zero_grad()
            loss.backward()
            optimi.step()
            train_losses.append(loss.data.item())

            if step % 300 == 0:
                print("=" * 60)
                print('Step: {}\t'
                  'Training Loss: {train_loss:.5f}\t'.format(
                    step, train_loss = sum(train_losses) / len(train_losses))
                )
                print("=" * 60)

            step = step + 1

        net.eval()
        with torch.no_grad():
            for inputs, outputs in tqdm(iter(val_loader)):
                inputs = inputs.to(device).float()
                outputs = outputs.to(device).float()
                preds = net(inputs)
                val_losses.append(loss_func(preds, outputs).data.item())

        print("=" * 60)
        print('Epoch: {}/{}\t'
          'Training Loss: {train_loss:.5f}\t'
          'Validation Loss: {val_loss:.5f}'.format(
            epoch + 1, 200, train_loss = sum(train_losses) / len(train_losses), 
            val_loss = sum(val_losses) / len(val_losses)))
        print("=" * 60)


train()