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
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class MassPrediction_XGBoost(MassPrediction):

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

        self.dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.x_test, label=self.y_test)
    def model_prepare(self):
        self.params = {
            'max_depth': 15,           # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
            'min_child_weight': 1,    # Minimum sum of instance weight (hessian) needed in a child.
            'eta': 0.3,               # Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features.
            'subsample': 1,           # Subsample ratio of the training instances.
            'colsample_bytree': 1,    # Subsample ratio of columns when constructing each tree.
            'objective': 'reg:squarederror', # Loss function to be minimized. In this case, regression with squared error.
            'eval_metric': 'rmse',    # Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification).
            'alpha': 0,               # L1 regularization term on weights. Increasing this value will make model more conservative.
            'lambda': 1,              # L2 regularization term on weights. Increasing this value will make model more conservative.
            'gamma': 0,               # Minimum loss reduction required to make a further partition on a leaf node of the tree.
            'seed': 42                # Random number seed. (Can be used for reproducibility.)
        }

    def train(self):
        num_boost_round = 100
        self.bst = xgb.train(self.params, self.dtrain, num_boost_round)

    def test(self):
        y_pred = self.bst.predict(self.dtest)


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