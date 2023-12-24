from data.data_loader import data_loader
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
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from MassPrediction import MassPrediction
class GNN(nn.Module):
    def __init__(self, input_size, num_hidden_layers=1,hidden_feature=256, output_size=1):
        super(GNN, self).__init__()

        layers = []

        # Graph Convolutional Layers
        self.layers_GCN1=GCNConv(input_size, hidden_feature,normalize=True)
        self.layers_GCN2=GCNConv(hidden_feature+input_size, hidden_feature,normalize=True)
        self.norm=nn.BatchNorm1d(hidden_feature)
        self.prelu=nn.PReLU()
        # Fully Connected Layers
        layers.append(nn.Linear(hidden_feature, hidden_feature))
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply graph convolutions
        x1 = self.layers_GCN1(x, edge_index)
        x1 = self.prelu(x1)
        x2 = self.layers_GCN2(torch.cat([x1, x], dim=1), edge_index)
        x2 = self.prelu(x2)
        # Apply fully connected layers
        out = self.layers(x2)

        return out


class MassPrediction_GNN(MassPrediction):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.data_prepare()
        self.model_prepare()

    def create_edge_index(self,data, scaler):
        # Extracting N and Z values

        data = scaler.inverse_transform(data).astype(int)
        data = torch.tensor(data)
        N = data[:, 0]
        Z = data[:, 1]

        # Broadcasting and condition check
        N_diff = torch.abs(N[:, None] - N[None, :])  # Difference in N between all pairs
        Z_diff = torch.abs(Z[:, None] - Z[None, :])  # Difference in Z between all pairs

        # Create mask for valid connections (N±1 and Z±1)
        mask = (N_diff <= 2) & (Z_diff <= 2)

        # Ensure no self-loops
        mask.fill_diagonal_(False)

        # Extract indices where mask is True
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()

        return edge_index

    def create_full_graph(self,X_train, X_test, y_train, y_test, scaler):
        # Combine train and test sets to form a complete graph
        X = torch.cat([torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float)], dim=0)
        y = torch.cat([torch.tensor(y_train, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)], dim=0)

        # Create edge indices for a fully connected graph
        # num_nodes = X.shape[0]
        # edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

        # create edge indices for N+-1 and Z+-1
        num_nodes = X.shape[0]
        edge_index = self.create_edge_index(X, scaler)

        # Create training and testing masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:len(X_train)] = True  # First part is training data
        test_mask[len(X_train):] = True  # Second part is test data

        data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
        return data
    def data_prepare(self):
        features, target = self.load_data()
        self.scaler_y = MinMaxScaler()
        self.target = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        self.scaler_x = MinMaxScaler()
        self.features = self.scaler_x.fit_transform(features)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.target,
                                                                                test_size=self.args.test_split,
                                                                                random_state=self.args.seed)
        self.graph_data = self.create_full_graph(self.x_train, self.x_test, self.y_train, self.y_test, self.scaler_x)
        self.test_sample_num = len(self.x_test)
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)
        self.graph_data.to(self.device)

    def model_prepare(self):
        self.model = GNN(input_size=self.x_train.shape[1]).to(self.device)

    def train(self):
        self.model.train()

        self.optimizer.zero_grad()
        train_mask = self.graph_data.train_mask

        output = self.model(self.graph_data)

        output = output.squeeze()
        loss = F.mse_loss(output[train_mask], self.graph_data.y[train_mask])

        loss.backward()
        self.optimizer.step()

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            test_mask = self.graph_data.test_mask
            output = self.model(self.graph_data).squeeze()
            test_loss = F.mse_loss(output[test_mask], self.graph_data.y[test_mask], reduction='sum').item()
            de_normalized_pred = self.scaler_y.inverse_transform(
                output[test_mask].detach().cpu().numpy().reshape(-1, 1)).flatten()
            de_normalized_target = self.scaler_y.inverse_transform(self.graph_data.y[test_mask].cpu().numpy().reshape(-1, 1)).flatten()
            differences = de_normalized_pred - de_normalized_target
            squared_diff = differences ** 2
            mean_squared_diff = np.mean(squared_diff)
            rms_error = np.sqrt(mean_squared_diff)
            rms_model = np.sqrt(np.mean(de_normalized_target ** 2))
        logging.info(
            f'Test Epoch {epoch}: Average loss: {test_loss:.4f} RMS: {rms_error:.4f}, model RMS: {rms_model:.4f}')
        return test_loss, rms_error

    def run(self):
        self.epoch_list = []
        self.loss_list = []
        best_RMS = 1e+6
        best_epoch = 0
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
            if test_loss_RMS_temp < best_RMS:
                best_epoch = epoch
                best_RMS = test_loss_RMS_temp
                if self.args.mode != "test":
                    torch.save(self.model.state_dict(),
                               self.args.save_dir + "./{}_{}.pth".format(self.args.method, self.args.data))
        logging.info(f'Training Finished: Best RMS: {best_RMS:.4f} at epoch {best_epoch}')

        self.draw_loss()
