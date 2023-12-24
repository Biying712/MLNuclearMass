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
from MassPrediction import MassPrediction
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

class MassPrediction_LightGBM(MassPrediction):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.data_prepare()
        self.model_prepare()

    def data_prepare(self):
        features, target = self.load_data()
        self.scaler_y = MinMaxScaler()
        self.target = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        self.scaler_x = MinMaxScaler()
        self.features = self.scaler_x.fit_transform(features)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.target,
                                                                                test_size=self.args.test_split,
                                                                                random_state=self.args.seed)
        self.test_sample_num = len(self.x_test)

    def model_prepare(self):
        params = {
            'boosting_type': 'gbdt',  # Traditional Gradient Boosting Decision Tree
            'objective': 'regression',  # Regression task
            'metric': 'l2',  # Evaluation metric for regression
            'num_leaves': 31,  # Number of leaves in one tree
            'learning_rate': 0.05,  # Learning rate
            'n_estimators': 50000,  # Number of boosting rounds
            'max_depth': -1,  # No limit on tree depth
            'subsample': 0.8,  # Subsample ratio of the training instance
            'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
            'min_child_weight': 0.001,  # Minimum sum of instance weight (hessian) needed in a child
            'reg_alpha': 0.0,  # L1 regularization term on weights
            'reg_lambda': 1.0,  # L2 regularization term on weights
            'max_bin': 255,  # Number of bins for feature discretization
            'min_data_in_leaf': 5  # Minimum number of data points in a leaf
        }

        # Initialize the LightGBM model with the defined hyperparameters
        self.model = lgb.LGBMRegressor(**params)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        y_pred = self.model.predict(self.x_test)

        # Calculating MSE loss
        test_loss = mean_squared_error(self.y_test, y_pred)

        # If a scaler is provided, denormalize the predictions and targets
        if self.scaler_y:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()

        # Calculate absolute errors
        abs_error = np.abs(y_pred - y_test)
        squre_error = abs_error ** 2
        rms = np.sqrt(squre_error.mean())

        rms_model=np.sqrt(np.mean(y_test**2))

        # Display results
        logging.info(f'Test: Average loss: {test_loss:.4f} RMS: {rms:.4f}, model RMS: {rms_model:.4f}')

        return test_loss, rms

    def run(self):
        self.train()
        test_loss,rms=self.test()
        logging.info(f'Training Finished: Best RMS: {rms:.4f}')

        # self.draw_loss()