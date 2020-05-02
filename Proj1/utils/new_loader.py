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
    
    def __init__(self,swap_channel) :
            
        train_input, train_target, train_classes, test_input, test_target,test_classes=load()
            
        #Data augmentation    
        if (swap_channel==True):
            
            train_input = torch.cat((train_input, train_input.flip(1)), dim=0)
            train_classes = torch.cat((train_classes, train_classes.flip(1)), dim=0)
            train_target = torch.cat((train_target, (train_classes.flip(1)[:,0] <= train_classes.flip(1)[:,1]).long()),dim=0)
        
        # Training set
        self.train_input  = train_input.sub(train_input.mean()).div(train_input.std())
        self.train_target  = train_target
        self.train_classes = train_classes
        
        #Test set 
        self.test_input  = test_input.sub(train_input.mean()).div(train_input.std())
        self.test_target = test_target
        self.test_classes = test_classes
       
               
    def __getitem__(self, index):
        
        return self.train_input[index], self.train_target[index], self.train_classes[index], self.test_input[index],  self.test_target[index],self.test_classes[index]


class Test_set(Dataset) :
    """
    Test set created from PairSetMNIST 
    """
    def __init__(self,PairSetMNIST) :
        
        self.len = PairSetMNIST.test_input.shape[0]
        self.test_input  = PairSetMNIST.test_input
        self.test_target  = PairSetMNIST.test_target
        self.test_classes = PairSetMNIST.test_classes
       
               
    def __getitem__(self, index):
        
        return self.test_input[index], self.test_target[index], self.test_classes[index]

    def __len__(self):

        return self.len

    
    
    
class Training_set (Dataset) :
    """
    Training set created from PairSetMNIST 
    Can split the training set in a 80 % training and 20 % validation
    """
    
    def __init__(self,PairSetMNIST) : 
        
        self.len = PairSetMNIST.train_input.shape[0]
        self.train_input =  PairSetMNIST.train_input
        self.train_target  = PairSetMNIST.train_target
        self.train_classes = PairSetMNIST.train_classes
            
        idx = list(range(len(PairSetMNIST.train_input)))
        split = int(np.floor(0.2 * len(PairSetMNIST.train_input)))
        train_idx = idx[split:]
        valid_idx = idx[:split]
            
        self.train_idx = train_idx
        self.valid_idx = valid_idx
            
    def __getitem__(self,index) : 
        
        return self.train_input[index], self.train_target[index], self.train_classes[index]
    
    def __len__(self):

        return self.len        
            
class Training_set_split(Dataset) :
    """
    Training set splitted  from Training set
    """
    
    def __init__(self,Training_set) :
        
        self.len = len(Training_set.train_idx)
        self.train_input =  Training_set.train_input[Training_set.train_idx]
        self.train_target  = Training_set.train_target[Training_set.train_idx]
        self.train_classes = Training_set.train_classes[Training_set.train_idx]
    
    def __getitem__(self,index) : 
        
        return self.train_input[index], self.train_target[index], self.train_classes[index]
    
    def __len__(self):

        return self.len
    
    
class Validation_set(Dataset) :
    """
    Validation set splitted  from Training set
    """
    
    def __init__(self,Training_set) :
        
        self.len = len(Training_set.valid_idx)
        self.valid_input =  Training_set.train_input[Training_set.valid_idx]
        self.valid_target  = Training_set.train_target[Training_set.valid_idx]
        self.valid_classes = Training_set.train_classes[Training_set.valid_idx]
    
    def __getitem__(self,index) : 
        
        return self.valid_input[index], self.valid_target[index], self.valid_classes[index]
    
    def __len__(self):

        return self.len