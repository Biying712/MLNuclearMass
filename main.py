from MassPrediction_NN import MassPrediction_NN
from MassPrediction_BNN import MassPrediction_BNN
from MassPrediction_LightGBM import MassPrediction_LightGBM
from MassPrediction_GNN import MassPrediction_GNN
from MassPrediction_MPNN import MassPrediction_MPNN
from MassPrediction_MDN import MassPrediction_MDN
from MassPrediction_GP import MassPrediction_GP
from MassPrediction_XGBoost import MassPrediction_XGBoost
import torch
import os
import argparse
import warnings
import logging
from datetime import datetime


parser = argparse.ArgumentParser(description='PyTorch MassPrediction Example')
parser.add_argument('method',
                    type=str,
                    choices={"NN","BNN","MDN","GNN", "MPNN", "LightGBM", "GP","XGB"},
                    help="ML method")
parser.add_argument('--data',
                    type=str,
                    choices={"AME","DZ","FRDM","HFB","WS4","ALL"},
                    help="name of the data file"
)
parser.add_argument('--batchsize',
                    type=int,
                    default=64,
                    metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr',
                    type=float,
                    default=0.1,
                    metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--scheduler_gamma',
                    type=float,
                    default=0.995,
                    metavar='M',
                    help='Learning rate scheduler step gamma (default: 0.995)')
parser.add_argument('--scheduler_step',
                    type=float,
                    default=100,
                    help='Learning rate scheduler step  (default: 100)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--save_dir',
                    type=str,
                    default='../model_saved')
parser.add_argument('--mode',
                    type=str,
                    choices={'train','test'},
                    required=True,
                    help='train | test')

parser.add_argument('--load_model',
                    type=str,
                    default="")
parser.add_argument('--num_monte_carlo',
                    type=int,
                    default=20,
                    metavar='N',
                    help='number of Monte Carlo samples to be drawn for inference')

parser.add_argument('--results_dir',
                    type=str,
                    default='../Loss_results',
                    metavar='N',
                    help=
                    'use tensorboard for logging and visualization of training progress')

parser.add_argument('--test_split',
                    type=float,
                    default=0.8,
                    help="the percentage of test set in all data")
parser.add_argument('--log_interval',
                    type=int,
                    default=20)
parser.add_argument('--log_dir',
                    type=str,
                    default='./log')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.001)

def config_logging():
    # Define the log filename with time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{args.log_dir}/training_{args.method}_{args.data}_split{args.test_split}_lr{args.lr}_wdecay{args.weight_decay}_schstep{args.scheduler_step}_epoch{args.epochs}_{current_time}.log"

    # Create a file handler for output to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

    # Create a stream handler for output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'))

    # Get the root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add the handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)



if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    config_logging()

    if args.method=="NN":
        MassPredictor=MassPrediction_NN(args)
    elif args.method=="BNN":
        MassPredictor=MassPrediction_BNN(args)
    elif args.method=="LightGBM":
        MassPredictor=MassPrediction_LightGBM(args)
    elif args.method=="GNN":
        MassPredictor=MassPrediction_GNN(args)
    elif args.method=="MPNN":
        MassPredictor=MassPrediction_MPNN(args)
    elif args.method=="MDN":
        MassPredictor=MassPrediction_MDN(args)
    elif args.method=="GP":
        MassPredictor=MassPrediction_GP(args)
    elif args.method=="XGB":
        MassPredictor=MassPrediction_XGBoost(args)
    else:
        raise NotImplementedError

    MassPredictor.run()


