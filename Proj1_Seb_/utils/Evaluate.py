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
 

def validate_model(Net,seed, mini_batch_size=100, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss(), n_epochs=40, 
                   eta=1e-3, lambda_l2 = 0, alpha=0.5, beta=0.5, plot=True,rotate = False,translate=False,
                   swap_channel = False,GPU=False): 

    """ 
    
    General :
         
         - Train a network model  which weights has been initialized with a specific seed over n_epochs 
         - Data is created with the same seed : train,validation and test calling the prologue
         - Record the train and validation accuracy and loss and can display they evolution curve
     
     
     Input :
     
         - Net : A network dictionnary from the <Nets> class
         - seed : seed for pseudo random number generator used in weight initialization and data loading
         -> mini_batch_size,optimizer, criterion, n_epochs, eta, lambda_2, alpha, beta see training.py
         - plot : if true plot the learning curve evolution over the epochs -> default true
         -> rotate,translate and swap_channels -> data augmentation see loader.py 
     
     Output : printed loss and accuracy of the network after training on the test set and learning curve if plot true
     
    """
    
    # set the pytorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # set the seed for random spliting of the dataset in training and validation
    random.seed(0)
    
    # create the dataset 
    data = PairSetMNIST()
    train_data = Training_set(data)
    test_data = Test_set(data)
    train_data_split =Training_set_split(train_data,rotate,translate,swap_channel)
    validation_data= Validation_set(train_data)
    
    
    # construct the net type with default parameter 
    if (Net['net_type'] == 'Net2c') :
        model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_prob = Net['drop_prob'])
    if (Net['net_type'] == 'LeNet_sharing') :
        model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_ws = Net['drop_prob_ws'],dropout_comp = Net['drop_prob_comp'])
    if (Net['net_type'] == 'LeNet_sharing_aux') :
        # check if any data augmentation has been called
        # if none construct with tuned parameters without data augmentation
        # if yes construct with tuned parameters with data augmentation
        if ( rotate == False and translate == False and swap_channel == False) :
            model = Net['net'](nbhidden_aux = Net['hidden_layers_aux'],nbhidden_comp = Net['hidden_layers_comp'],
                               drop_prob_aux =Net['drop_prob_aux'],drop_prob_comp = Net['drop_prob_comp'])
        else :
            Net['learning rate'] = Net['learning rate augm']
            model = Net['net'](nbhidden_aux = Net['hidden_layers_aux'],nbhidden_comp = Net['hidden_layers_comp'],
                               drop_prob_aux =Net['drop_prob_aux_augm'],drop_prob_comp = Net['drop_prob_comp_augm'])       
    if (Net['net_type'] == 'Google_Net') :
        model = Net['net'](channels_1x1 = Net['channels_1x1'],
                           channels_3x3 = Net['channels_3x3'],channels_5x5=Net['channels_5x5'],pool_channels = Net['pool_channels'],
                           nhidden = Net['hidden_layers'], drop_prob_comp = Net['drop_prob_comp'],drop_prob_aux = Net['drop_prob_aux'])
        

    if GPU and cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    model =model.to(device)
    
    # train the model on the train set and validate at each epoch    
    train_losses, train_acc, valid_losses, valid_acc = train_model(model, train_data_split, validation_data, device, mini_batch_size,
                                                                   optimizer,criterion,n_epochs, Net['learning rate'],lambda_l2,
                                                                   alpha, beta)
    
    if plot:
        
        learning_curve(train_losses, train_acc, valid_losses, valid_acc)
    
    # loss and accuracy of the network on the test
    test_loss, test_accuracy = compute_metrics(model, test_data, device)
    
    print('\nTest Set | Loss: {:.4f} | Accuracy: {:.2f}%\n'.format(test_loss, test_accuracy))
    

########################################################################################################################################

# evaluation and final prediction statistics on test set

def evaluate_model(Net, seeds, mini_batch_size=100, optimizer = optim.Adam, criterion = nn.CrossEntropyLoss(), n_epochs=40, eta = 1e-3,
                   lambda_l2 = 0, alpha=0.5, beta=0.5, plot=True,statistics = True ,rotate = False,translate=False,swap_channel = False,
                   GPU=False): 
    
    """ 
    General : 10 rounds of network training / validation with statistics
         
         - Repeat the training/validation procedure 10 times for ten different seeds in seeds
             1) At every seed -> reinitializes a network and a dataset : train,validation and test 
             2) Weights initialization and data loading are using the seed 
             3) Record the train and validation accuracy and loss and can display their evolution curve
             4) Compute the statistics at the end of each training for performance evaluation
                 i)  Mean training accuracy for each seed -> value at the end of the last epoch
                 ii) Mean validation accuracy for each seed -> value at the end of the last epoch
                 iii) Mean test accuracy for each seed -> compute the accuracy on the test after each training
                 -> display a boxplot of the statistics if statistics is true and print the mean and standard deviation
     
     Input :
     
         - Net : A network dictionnary from the <Nets> class
         - seeds : a list of seed to iterate over for pseudo random number generator used in weight initialization and data loading
         -> mini_batch_size,optimizer, criterion, n_epochs, eta, lambda_2, alpha, beta see training.py
         - plot : if true plot the learning curve evolution over the epochs -> default true
         - statistics : if true display the boxplot of the train accuracies, validations and test and print the mean and standard deviation 
                        statistics
         -> rotate,translate and swap_channels -> data augmentation see loader.py 
     
     Output : 
     
         - train_result : A (10x4xn_epochs) tensor 
                             10 -> seed
                             4 -> train loss ,train accuracy, validation loss, validation accuracy
                             n_epochs -> evolution during training
         - test_losses : A tensor of shape (10,) containing the test loss at each seed
         - test_accuracies : A tensor of shape (10,) containing the test loss at each seed
         
    """
    
    # tensor initialization to store the metrics
    train_results = torch.empty(len(seeds), 4, n_epochs)
    test_losses = []
    test_accuracies = []
    
    for n, seed in enumerate(seeds):
        
        # set the pytorch seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) 
        
        # set the seed for random spliting of the dataset in training and validation
        random.seed(0)
        
        # load the dataset train,validation and test
        data = PairSetMNIST()
        train_data = Training_set(data)
        test_data = Test_set(data)
        train_data_split =Training_set_split(train_data,rotate,translate,swap_channel)
        validation_data= Validation_set(train_data)
        
        # construct the net type with default parameter 
        if (Net['net_type'] == 'Net2c') :
            model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_prob = Net['drop_prob'])
        if (Net['net_type'] == 'LeNet_sharing') :
            model = Net['net'](nb_hidden = Net['hidden_layers'],dropout_ws = Net['drop_prob_ws'],dropout_comb = Net['drop_prob_comp'])
        if (Net['net_type'] == 'LeNet_sharing_aux') :
            # check if any data augmentation has been called
            # if none construct with tuned parameters without data augmentation
            # if yes construct with tuned parameters with data augmentation
            if (  rotate == False and translate == False and swap_channel == False) :
                model = Net['net'](nbhidden_aux = Net['hidden_layers_aux'],nbhidden_comp = Net['hidden_layers_comp'],
                                   drop_prob_aux =Net['drop_prob_aux'],drop_prob_comp = Net['drop_prob_comp'])
            else :
                Net['learning rate'] = Net['learning rate augm']
                model = Net['net'](nbhidden_aux = Net['hidden_layers_aux'],nbhidden_comp = Net['hidden_layers_comp'],
                                   drop_prob_aux =Net['drop_prob_aux_augm'],drop_prob_comp = Net['drop_prob_comp_augm'])       
        if (Net['net_type'] == 'Google_Net') :
            model = Net['net'](channels_1x1 = Net['channels_1x1'],
                               channels_3x3 = Net['channels_3x3'],channels_5x5=Net['channels_5x5'],pool_channels = Net['pool_channels'],
                               nhidden = Net['hidden_layers'], drop_prob_comp = Net['drop_prob_comp'],drop_prob_aux = Net['drop_prob_aux'])
         

        if GPU and cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')


        model = model.to(device)
        
        # train the model on the train set and validate at each epoch 
        train_losses, train_acc, valid_losses, valid_acc = train_model(model, train_data_split, validation_data, device, mini_batch_size,
                                                                       optimizer,criterion,n_epochs, Net['learning rate'],lambda_l2,
                                                                       alpha, beta)
        # store the training and validation accuracies and losses during the training 
        train_results[n,] = torch.tensor([train_losses, train_acc, valid_losses, valid_acc])
        # compute the loss and accuracy of the model on the test set
        test_loss, test_acc = compute_metrics(model, test_data, device)
        # store the test metrics in the list
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # learning curve
        if plot:
            learning_curve(train_losses, train_acc, valid_losses, valid_acc)
        
        print('Seed {:d} | Test Loss: {:.4f} | Test Accuracy: {:.2f}%\n'.format(n, test_loss, test_acc))
    
    # store the train, validation and test accuracies in a tensor for the boxplot
    data = torch.stack([train_results[:,1,(n_epochs-1)], train_results[:,3,(n_epochs-1)] , torch.tensor(test_accuracies)])
    
    # boxplot
    if statistics :
        Title = Net['net_type'] + " Accuracies"
        boxplot(data, Title)

    return train_results, torch.tensor(test_losses), torch.tensor(test_accuracies)

#######################################################################################################################################