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

###########################################################################################################################################

class Nets () :
    
    """
    
     General :
     
      - A class containing dictionnaries of the Nets used in this project which are defined by default with the parameters we tuned on a 
        certain grid
      
      - Each dictionnaries contain the name, an uninitialized instance, the learning rate and the parameters used in the constructor of the 
        network
     
     Parameters :
      
      1) Net2c -> Basic CNN for binary classification 
      2) LeNet_sharing -> LeNet inspired CNN applied to each channel (weight sharing)  then concatenation for binary
      3) LeNet_sharing_aux -> LeNet inspired CNN applied to each channel (weight sharing) and used of auxiliary loss then concatenation for 
         binary -> contain parameters for a model without and with data augmentation
      4) Google_Net -> Inception bloc over the two channel followed by a CNN for auxiliary loss (weight sahring) then concatenation for 
         binary
         
    Functions :
    
      1) Tune_Net2c : a function called to tune the parameters of the Net2c model which call a grid search function specific for Net2c from
         grid_search.py
          
          - lrs : learning rate list of value to tune 
          - drop_prob : dropout rate value list to tune
          - hidden_layers : hidden layers list of value to tune
          - seeds : list of seed value used in the grid search function for reproducible statistics
          -> rest of parameters are default values used in the grid search 
          
      2) Tune_LeNet_sharing : a function called to tune the parameters of the LeNet_sharing model which call a grid search function 
         specific for LeNet_sharing from grid_search.py
          
          - lrs : learning rate list of value to tune 
          - drop_prob_ws : dropout rate list of value of the FC layers of the CNN   to tune
          - drop_prob_comp : dropout rate list of value of the FC layers of binary classification  to tune
          - seeds : list of seed value used in the grid search function for reproducible statistics
          -> rest of parameters are default values used in the grid search   
     
      3) Tune_LeNet_sharing_aux : a function called to tune the parameters of the LeNet_sharing_aux model which call a grid search function 
         specific for LeNet_sharing_aux from grid_search.py
          
          - lrs : learning rate list of value to tune 
          - drop_prob_aux : dropout rate list of value of the auxiliary CNN
          - drop_prob_comp : dropout rate list of value of the FC layers of binary classification  to tune
          - seeds : list of seed value used in the grid search function for reproducible statistics
          -> rest of parameters are default values used in the grid search
          -> differentiate if the model was tuned with data augmentation or not
          
       => In each case change the value in the dictionnary of the model that has been tuned by the optimals for the current instance
       => To save them -> change thevalue in the constructor by the optimal value which has been printed at the end of the grid search
       
    """
    
    # constructor
    def __init__(self):
        
        super(Nets, self).__init__()
  
        self.Net2c = {'net_type' : 'Net2c', 'net' : Net2C , 'learning rate' : 0.001, 'hidden_layers': 150, 'drop_prob' :0.2 }
        self.LeNet_sharing = {'net_type' : 'LeNet_sharing', 'net' : LeNet_sharing , 'learning rate' : 0.001, 'hidden_layers': 100,
                              'drop_prob_ws':0.5,'drop_prob_comp': 0.2}
        self.LeNet_sharing_aux = {'net_type' : 'LeNet_sharing_aux', 'net' : LeNet_sharing_aux , 'learning rate' : 0.01,
                                  'learning rate augm': 0.01,'hidden_layers_aux':200,'hidden_layers_comp':60,'drop_prob_aux':0.4,
                                  'drop_prob_comp': 0.1,'drop_prob_aux_augm':0.3,'drop_prob_comp_augm': 0.1}
        self.Google_Net = {'net_type' : 'Google_Net', 'net' : Google_Net, 'learning rate' : 0.001,'channels_1x1' : 64,
                           'channels_3x3' : 64,'channels_5x5' : 64,'pool_channels' : 64,'hidden_layers':200,
                           'drop_prob_comp':0,'drop_prob_aux': 0.7}
    # tuning function
    def Tune_Net2c(self,lrs,drop_prob, hidden_layers,seeds,mini_batch_size=100, optimizer = optim.Adam,criterion = nn.CrossEntropyLoss(),
                   n_epochs=40, lambda_l2 = 0,alpha = 0.5, beta = 0.5, rotate =False,translate=False,swap_channel = False, GPU=False) :
        
        # Call the grid search function
        train_results, test_losses, test_accuracies,opt_lr, opt_prob, opt_hidden_layer = grid_search_basic(lrs,drop_prob, hidden_layers, seeds,mini_batch_size, optimizer ,criterion , n_epochs, lambda_l2 ,alpha,beta, rotate ,translate,swap_channel , GPU)
        
        # save the optimal value in the dictionnary for the current instance
        self.Net2c['learning rate'] = opt_lr
        self.Net2c['drop_prob'] = opt_prob
        self.Net2c['hidden_layers'] = opt_hidden_layer
        
        return train_results, test_losses, test_accuracies
    
    def Tune_LeNet_sharing (self,lrs,drop_prob_ws, drop_prob_comp,seeds,mini_batch_size=100, optimizer = optim.Adam,
                            criterion = nn.CrossEntropyLoss(),n_epochs=40, lambda_l2 = 0,alpha = 0.5, beta = 0.5, 
                            rotate =False,translate=False, swap_channel = False, GPU=False):
        
        # Call the grid search function
        train_results, test_losses, test_accuracies,opt_lr, opt_prob_ws, opt_prob_comp = grid_search_ws(lrs,drop_prob_ws, drop_prob_comp, seeds, mini_batch_size, optimizer ,criterion , n_epochs, lambda_l2 ,alpha, beta, rotate ,translate, swap_channel , GPU)
        
        # save the optimal value in the dictionnary for the current instance
        self.LeNet_sharing['learning rate'] = opt_lr
        self.LeNet_sharing['drop_prob_aux'] = opt_prob_ws
        self.LeNet_sharing['drop_prob_comp'] = opt_prob_comp
        
        return train_results, test_losses, test_accuracies
    
    def Tune_LeNet_sharing_aux (self,lrs,drop_prob_aux, drop_prob_comp,seeds,mini_batch_size=100, optimizer = optim.Adam,
                                criterion = nn.CrossEntropyLoss(),n_epochs=40, lambda_l2 = 0, alpha = 0.5, beta = 0.5,  
                                rotate =False,translate=False, swap_channel = False, GPU=False):
        
        # Call the grid search function
        train_results, test_losses, test_accuracies,opt_lr, opt_prob_aux, opt_prob_comp = grid_search_aux(lrs,drop_prob_aux, drop_prob_comp, seeds, mini_batch_size, optimizer,criterion,n_epochs,lambda_l2 , alpha, beta,rotate,translate, swap_channel, GPU)
        
        # save the optimal value in the dictionnary for the current instance
        if (rotate == True or translate == True or swap_channel == True) :
            self.LeNet_sharing_aux['learning rate augm'] = opt_lr
            self.LeNet_sharing_aux['drop_prob_aux_augm'] = opt_prob_aux
            self.LeNet_sharing_aux['drop_prob_comp_augm'] = opt_prob_comp
            
        else :
            self.LeNet_sharing_aux['learning rate'] = opt_lr
            self.LeNet_sharing_aux['drop_prob_aux'] = opt_prob_aux
            self.LeNet_sharing_aux['drop_prob_comp'] = opt_prob_comp
        
        return train_results, test_losses, test_accuracies

###########################################################################################################################################
