import torch
import numpy as np
import utils.dlc_practical_prologue as prologue
from torch.utils.data import Dataset, DataLoader




def load():
    '''Load the data in the format required by the project
    
        Returns: tuple
        
        tuple[0]: train
        tuple[1]:target
        tuple[2]: classes
    '''
    return prologue.generate_pair_sets(1000)



class PairSetMNIST(Dataset):
    
    """
    Generate train input, labels, and classes set from load()
    
    """
    
    def __init__(self, train=False, test=False, valid=False): # add swap_channels
        
        train_input, train_target, train_classes, test_input,test_target,test_classes=load()
        
        train_input =train_input.sub(train_input.mean()).div(train_input.std())
        
        idx = list(range(len(train_input)))
        split = int(np.floor(0.2 * len(train_input)))
        train_idx = idx[split:]
        valid_idx = idx[:split]
        
        if train:
    
            self.len = train_input[train_idx].shape[0]
            self.x_  = train_input[train_idx]
            self.y_  = train_target[train_idx]
            self.classes_ = train_classes[train_idx]
            
        if valid:
            
            self.len = train_input[valid_idx].shape[0]
            self.x_ = train_input[valid_idx]
            self.y_ = train_target[valid_idx]
            self.classes_ = train_classes[valid_idx]
        
        elif test:
            
            self.len = test_input.shape[0]
            self.x_  = test_input.sub(train_input.mean()).div(train_input.std())
            self.y_  = test_target
            self.classes_ = test_classes
            
               
    def __getitem__(self, index):
        
        return self.x_[index], self.y_[index], self.classes_[index]

    def __len__(self):

        return self.len     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
