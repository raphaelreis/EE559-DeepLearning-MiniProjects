import sys
sys.path.append('..')
import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
import torch.utils.data as dt
from models.Inception_Net import Google_Net
from models.Basic import Net2C
from models.Le_Net import LeNet_sharing_aux,LeNet_sharing
from utils.grid_search import grid_search_basic,grid_search_ws,grid_search_aux

class Nets () :
    
    def __init__(self):
        
        super(Nets, self).__init__()
  
        self.Net2c = {'net_type' : 'Net2c', 'net' : Net2C , 'learning rate' : 0.001, 'hidden_layers': 100, 'drop_prob' :0.0 }
        self.LeNet_sharing = {'net_type' : 'LeNet_sharing', 'net' : LeNet_sharing , 'learning rate' : 0.001, 'hidden_layers': 100,
                              'drop_prob_ws':0.4,'drop_prob_comp': 0.3}
        self.LeNet_sharing_aux = {'net_type' : 'LeNet_sharing_aux', 'net' : LeNet_sharing_aux , 'learning rate' : 0.01,
                                  'learning rate augm': 0.01,'hidden_layers_aux':200,'hidden_layers_comp':60,'drop_prob_aux':0.4,
                                  'drop_prob_comp': 0.1,'drop_prob_aux_augm':0.2,'drop_prob_comp_augm': 0.05}
        self.Google_Net = {'net_type' : 'Google_Net', 'net' : Google_Net, 'learning rate' : 0.001,'channels_1x1' : 64,
                           'channels_3x3' : 64,'channels_5x5' : 64,'pool_channels' : 64,'hidden_layers':200,
                           'drop_prob_comp':0,'drop_prob_aux': 0.7}
    
    def Tune_Net2c(self,lrs,drop_prob, hidden_layers,seeds,mini_batch_size=100, optimizer = optim.Adam,criterion = nn.CrossEntropyLoss(),
                   n_epochs=40, lambda_l2 = 0, rotate =False,translate=False,swap_channel = False, GPU=False) :
        
        train_results, test_losses, test_accuracies,opt_lr, opt_prob, opt_hidden_layer = grid_search_basic(lrs,drop_prob, hidden_layers, seeds,mini_batch_size=100, optimizer = optim.Adam,criterion = nn.CrossEntropyLoss(), n_epochs=40, lambda_l2 = 0, rotate =False,translate=False,swap_channel = False, GPU=False)
        
        self.Net2c['learning rate'] = opt_lr
        self.Net2c['drop_prob'] = opt_prob
        self.Net2c['hidden_layers'] = opt_hidden_layer
        
        return train_results, test_losses, test_accuracies
    
    def Tune_LeNet_sharing (self,lrs,drop_prob_ws, drop_prob_comp,seeds,mini_batch_size=100, optimizer = optim.Adam,
                            criterion = nn.CrossEntropyLoss(),n_epochs=40, lambda_l2 = 0, rotate =False,translate=False,
                            swap_channel = False, GPU=False):
        train_results, test_losses, test_accuracies,opt_lr, opt_prob_ws, opt_prob_comp = grid_search_ws(lrs,drop_prob_ws, drop_prob_comp, seeds, mini_batch_size=100, optimizer = optim.Adam,criterion = nn.CrossEntropyLoss(), n_epochs=40, lambda_l2 = 0,alpha=0.5, beta=0.5, rotate = False,translate=False, swap_channel = False, GPU=False)
        
        self.LeNet_sharing['learning rate'] = opt_lr
        self.LeNet_sharing['drop_prob_aux'] = opt_prob_ws
        self.LeNet_sharing['drop_prob_comp'] = opt_prob_comp
        
        return train_results, test_losses, test_accuracies
    
    def Tune_LeNet_sharing_aux (self,lrs,drop_prob_aux, drop_prob_comp,seeds,mini_batch_size=100, optimizer = optim.Adam,
                                criterion = nn.CrossEntropyLoss(),n_epochs=40, lambda_l2 = 0, rotate =False,translate=False,
                                swap_channel = False, GPU=False):
        
        train_results, test_losses, test_accuracies,opt_lr, opt_prob_aux, opt_prob_comp = grid_search_aux(lrs,drop_prob_aux, drop_prob_comp, seeds, mini_batch_size=100, optimizer = optim.Adam,criterion= nn.CrossEntropyLoss(),n_epochs=40,lambda_l2 = 0, alpha=0.5, beta=0.5,rotate=False,translate=False, swap_channel = False, GPU=False)
        
        if (rotate == True or translate == True or swap_channel == True) :
            self.LeNet_sharing_aux['learning rate augm'] = opt_lr
            self.LeNet_sharing_aux['drop_prob_aux_augm'] = opt_prob_aux
            self.LeNet_sharing_aux['drop_prob_comp_augm'] = opt_prob_comp
            
        else :
            self.LeNet_sharing_aux['learning rate'] = opt_lr
            self.LeNet_sharing_aux['drop_prob_aux'] = opt_prob_aux
            self.LeNet_sharing_aux['drop_prob_comp'] = opt_prob_comp
        
        return train_results, test_losses, test_accuracies
