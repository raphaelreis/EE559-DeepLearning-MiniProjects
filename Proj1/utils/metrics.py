import torch 
import numpy as np
from torch import nn 
from torch.nn import functional as F
from torch import optim
import torch.utils.data as dt
from torch.utils.data import Dataset, DataLoader
import torch.cuda as cuda



def accuracy(y_pred, target):
    """ 
    Return the accuracy of a prediction given the target 
    
    Input : 
        
        1) y_pred : vector of predicted value
        2) target : vector or target value
        
    Output : Accuracy -> number of True predictions in  percentage 0-100%
        
    """

    assert y_pred.shape[0] == len(target), "y_pred and target should be the same shape"

    return (y_pred.argmax(1) == target).sum().float() / float(target.shape[0])

##############################################################################################################

def compute_nb_errors(y_pred, target):
    
    """ 
    Return the number of errors of a prediction given the target
    
    Input : 
        
        1) y_pred : vector of predicted value
        2) target : vector or target value
        
    Output : number of errors -> number of False predicted value
    """
    
    assert y_pred.shape[0] == len(target), "y_pred and target should be the same shape"
    
    return float(len(target) - (y_pred.argmax(1) == target).sum())

##############################################################################################################

def compute_metrics(model, Data, device, mini_batch_size=100, criterion = nn.CrossEntropyLoss()):
    
    """
    Function to calculate prediction accuracy and loss of a model on a data
    
    Input : 
    
        - model : an instance of a neural network class
        - data : Dataset class from pytorch with the data : input,target,class
        
    Output :
    
        - Accuracy of the model on the data
        - loss of the model given the criterion
    
    """
    # data loader from pytorch feed with the data
    data_loader = DataLoader(Data, batch_size=mini_batch_size, shuffle=True)
    model.eval()
    test_loss = 0
    nb_errors = 0
    
    with torch.no_grad():
        
        for i, data in enumerate(data_loader, 0):
            
            input_, target_, classes_ = data

            input_ = input_.to(device)
            target_ = target_.to(device)
            classes_ = classes_.to(device)
            
            # check which kind of net it is 
            if (model.__class__.__name__ == 'LeNet_sharing_aux' or model.__class__.__name__ == 'Google_Net' ) :
                _, _, output = model(input_) 
            else :
                output = model(input_)
            
            # compute loss and the number of erros on the batch and add the loss to the overall loss on the data
            batch_loss = criterion(output, target_)
            test_loss += batch_loss     
            nb_errors += compute_nb_errors(output, target_)
        
        # compute accuracy and loss   
        acc = 100*(1 - (nb_errors/Data.len) )
        test_loss = test_loss/Data.len     # normalize loss
              
        return test_loss.item(), acc