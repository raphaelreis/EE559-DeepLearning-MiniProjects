import torch 
import matplotlib.pyplot as plt
import sys
import random
import numpy as np
sys.path.append('..')
from torch import nn 
from torch.nn import functional as F
from torch import optim
import torch.utils.data as dt
from torch.utils.data import Dataset, DataLoader
from utils.loader import load,PairSetMNIST,Training_set,Test_set, Training_set_split,Validation_set
from utils.plot import learning_curve, boxplot
from utils.metrics import accuracy, compute_nb_errors, compute_metrics
from utils.training import train_model
import torch.cuda as cuda

# simple validation 

def validate_model(Net,seed, mini_batch_size=100, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss(), n_epochs=40, 
                   eta=1e-3, lambda_l2 = 0, alpha=0.5, beta=0.5, plot=True,rotate = False,translate=False,
                   swap_channel = False,GPU=False): 

    """ Training / validation over n_epochs + testing a full test set"""
    
    # set the pytorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # set the seed for random spliting of the dataset in training and validation
    random.seed(0)
    
    
    data = PairSetMNIST( rotate,translate,swap_channel)
    train_data = Training_set(data)
    test_data = Test_set(data)
    train_data_split =Training_set_split(train_data)
    validation_data= Validation_set(train_data)

    if (Net['net_type'] == 'Net2c') :
        model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_prob = Net['drop_prob'])
    if (Net['net_type'] == 'LeNet_sharing') :
        model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_ws = Net['drop_prob_ws'],dropout_comb = Net['drop_prob_comb'])
    if (Net['net_type'] == 'LeNet_sharing_aux') :
        model = Net['net'](nbhidden_aux = Net['hidden_layers_aux'],nbhidden_comp = Net['hidden_layers_comp'],
                           drop_prob_aux =Net['drop_prob_aux'],drop_prob_comp = Net['drop_prob_comb'])
    if (Net['net_type'] == 'Google_Net') :
        model = Net['net'](channels_1x1 = Net['channels_1x1'],
                           channels_3x3 = Net['channels_3x3'],channels_5x5=Net['channels_5x5'],pool_channels = Net['pool_channels'],
                           nhidden = Net['hidden_layers'], drop_prob_comp = Net['drop_prob_comb'],drop_prob_aux = Net['drop_prob_aux'])
        

    if GPU and cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    model =model.to(device)
        
    train_losses, train_acc, valid_losses, valid_acc = train_model(model, train_data_split, validation_data, device, mini_batch_size,
                                                                   optimizer,criterion,n_epochs, Net['learning rate'],lambda_l2,
                                                                   alpha, beta)
    
    if plot:
        
        learning_curve(train_losses, train_acc, valid_losses, valid_acc)

    test_loss, test_accuracy = compute_metrics(model, test_data, device)
    
    print('\nTest Set | Loss: {:.4f} | Accuracy: {:.2f}%\n'.format(test_loss, test_accuracy))
    

########################################################################################################################################

# evaluation and final prediction statistics on large test set

def evaluate_model(Net, seeds, mini_batch_size=100, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss(), n_epochs=40, eta = 1e-3,
                   lambda_l2 = 0, alpha=0.5, beta=0.5, plot=True,rotate = False,translate=False,swap_channel = False, GPU=False): 
    
    """ 10 rounds of training / validation + testing metrics statistics  """
    
    train_results = torch.empty(len(seeds), 4, n_epochs)
    test_losses = []
    test_accuracies = []
    
    for n, seed in enumerate(seeds):
        
        # set the pytorch seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) 
        
        # set the seed for random spliting of the dataset in training and validation
        random.seed(0)
        
        data = PairSetMNIST(rotate,translate,swap_channel)
        train_data = Training_set(data)
        test_data = Test_set(data)
        train_data_split =Training_set_split(train_data)
        validation_data= Validation_set(train_data)
        
        if (Net['net_type'] == 'Net2c') :
            model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_prob = Net['drop_prob'])
        if (Net['net_type'] == 'LeNet_sharing') :
            model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_ws = Net['drop_prob_ws'],dropout_comb = Net['drop_prob_comb'])
        if (Net['net_type'] == 'LeNet_sharing_aux') :
            model = Net['net'](nbhidden_aux = Net['hidden_layers_aux'],nbhidden_comp = Net['hidden_layers_comp'],
                               drop_prob_aux =Net['drop_prob_aux'],drop_prob_comp = Net['drop_prob_comb'])
        if (Net['net_type'] == 'Google_Net') :
            model = Net['net'](channels_1x1 = Net['channels_1x1'],
                               channels_3x3 = Net['channels_3x3'],channels_5x5=Net['channels_5x5'],pool_channels = Net['pool_channels'],
                               nhidden = Net['hidden_layers'], drop_prob_comp = Net['drop_prob_comb'],drop_prob_aux = Net['drop_prob_aux'])
         

        if GPU and cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')


        model = model.to(device)

        train_losses, train_acc, valid_losses, valid_acc = train_model(model, train_data_split, validation_data, device, mini_batch_size,
                                                                       optimizer,criterion,n_epochs, Net['learning rate'],lambda_l2,
                                                                       alpha, beta)
        train_results[n,] = torch.tensor([train_losses, train_acc, valid_losses, valid_acc])
        test_loss, test_acc = compute_metrics(model, test_data, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if plot:
            
            learning_curve(train_losses, train_acc, valid_losses, valid_acc)
            
            
        
        print('Seed {:d} | Test Loss: {:.4f} | Test Accuracy: {:.2f}%\n'.format(n, test_loss, test_acc))

    return train_results, torch.tensor(test_losses), torch.tensor(test_accuracies)

#######################################################################################################################################