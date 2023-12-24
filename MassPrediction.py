import torch
from matplotlib import pyplot as plt
from data.data_loader import data_loader

class MassPrediction():
    def __init__(self,args):
        self.data=args.data
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
    def load_data(self):
        AME_features,AME_mass,AME_mass_excess,DZ_features,DZ_residual,FRDM_features,FRDM_residual,HFB_features,HFB_residual,WS4_features,WS4_residual=data_loader()
        if self.data=="AME":
            return AME_features,AME_mass_excess
        elif self.data=="DZ":
            return DZ_features,DZ_residual
        elif self.data=="FRDM":
            return FRDM_features,FRDM_residual
        elif self.data=="HFB":
            return HFB_features,HFB_residual
        elif self.data=="WS4":
            return WS4_features,WS4_residual
    def draw_loss(self):

        plt.plot(self.epoch_list, self.loss_list, marker='*')
        plt.title('Loss Function Plot')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['RMS'])
        plt.grid(True)
        plt.savefig('{}/{}_{}_testsplit_{}.png'.format(self.args.results_dir, self.args.method, self.args.data, self.args.test_split))
        plt.show()