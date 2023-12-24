from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import BatchSampler
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging
from MassPrediction import MassPrediction
class NN(nn.Module):
    def __init__(self, input_size,num_hidden_layers=8,hidden_feature=512,output_size=1):
        super(NN, self).__init__()

        layers = []

        # First layer (input layer to first hidden layer)
        layers.append(nn.Linear(input_size, hidden_feature))
        layers.append(nn.BatchNorm1d(hidden_feature))
        layers.append(nn.PReLU())

        # Add additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_feature, hidden_feature))
            layers.append(nn.BatchNorm1d(hidden_feature))
            layers.append(nn.PReLU())

        # Output layer
        layers.append(nn.Linear(hidden_feature, output_size))

        # Combine all layers into a Sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MassPrediction_NN(MassPrediction):
    
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.data_prepare()
        self.model_prepare()
    def data_prepare(self):
        features,target=self.load_data()
        self.scaler_y = MinMaxScaler()
        self.target = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        self.scaler_x = MinMaxScaler()
        self.features = self.scaler_x.fit_transform(features)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.target, test_size=self.args.test_split, random_state=self.args.seed)
        self.test_sample_num=len(self.x_test)
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)
    def model_prepare(self):
        self.model = NN(input_size=self.x_train.shape[1]).to(self.device)
    def train(self):
        self.model.train()
        sampler = BatchSampler(range(len(self.x_train)), batch_size=self.args.batchsize, drop_last=False)
        for batch_indices in sampler:
            data = self.x_train[batch_indices]
            target = self.y_train[batch_indices]
            self.optimizer.zero_grad()

            output = self.model(data)

            loss = F.mse_loss(output[:, 0], target)

            loss.backward()
            self.optimizer.step()
    def test(self,epoch):
        self.model.eval()
        with torch.no_grad():

            data = self.x_test
            target = self.y_test
            output = self.model(data)
            test_loss = F.mse_loss(output[:, 0], target, reduction='mean').item()
            de_normalized_pred = self.scaler_y.inverse_transform(output[:, 0].detach().cpu().numpy().reshape(-1, 1)).flatten()
            de_normalized_target = self.scaler_y.inverse_transform(target.cpu().numpy().reshape(-1, 1)).flatten()
            differences = de_normalized_pred - de_normalized_target
            squared_diff = differences ** 2
            mean_squared_diff = np.mean(squared_diff)
            rms_error = np.sqrt(mean_squared_diff)
            rms_model=np.sqrt(np.mean(de_normalized_target**2))
        logging.info(f'Test Epoch {epoch}: Average loss: {test_loss:.4f} RMS: {rms_error:.4f}, model RMS: {rms_model:.4f}')
        return test_loss, rms_error
    def run(self):
        self.epoch_list=[]
        self.loss_list=[]
        best_RMS=1e+6
        best_epoch=0
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay,
                                   momentum=0.9)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)

        for epoch in range(1, self.args.epochs + 1):
            if self.args.mode != "test":
                self.train()
            test_loss_temp, test_loss_RMS_temp = self.test(epoch)
            self.scheduler.step()
            self.epoch_list.append(epoch)
            self.loss_list.append(test_loss_temp)
            if test_loss_RMS_temp<best_RMS:
                best_epoch=epoch
                best_RMS=test_loss_RMS_temp
                if self.args.mode != "test":
                    torch.save(self.model.state_dict(), self.args.save_dir + "./{}_{}.pth".format(self.args.method, self.args.data))
        logging.info(f'Training Finished: Best RMS: {best_RMS:.4f} at epoch {best_epoch}')

        self.draw_loss()
