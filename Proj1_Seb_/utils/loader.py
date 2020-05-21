import torch
import numpy as np
import random
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

####################################################################################################################3

class PairSetMNIST(Dataset):
    
    """
    Generate train input, labels, and classes set from load()
    Data can be augmented in three ways: 
        1) Rotatation of the digits by 90,180 and 270 degrees except for digits 6 and 9 to avoid confusion
        2) translation one pixel upward,downward,left and right
        3) swap the order of the two digits channels
    Everything can be done together or individually
    """
    
    def __init__(self) :
            
        train_input, train_target, train_classes, test_input, test_target,test_classes=load()
        
        train_input = train_input.sub(train_input.mean()).div(train_input.std())
        
        # Training set
        self.train_input  = train_input
        self.train_target  = train_target
        self.train_classes = train_classes
        
        #Test set 
        self.test_input  = test_input.sub(test_input.mean()).div(test_input.std())
        self.test_target = test_target
        self.test_classes = test_classes
       
               
    def __getitem__(self, index ):
        
        return self.train_input[index], self.train_target[index], self.train_classes[index], self.test_input[index],                       self.test_target[index],self.test_classes[index]

#################################################################################################################################

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

#################################################################################################################################
    
class Training_set (Dataset) :
    """
    Training set created from PairSetMNIST 
    Can split the training set in a 80 % training and 20 % validation randomly
    """
    
    def __init__(self,PairSetMNIST) : 
        
        self.len = PairSetMNIST.train_input.shape[0]
        self.train_input =  PairSetMNIST.train_input
        self.train_target  = PairSetMNIST.train_target
        self.train_classes = PairSetMNIST.train_classes
            
        idx = list(range(len(PairSetMNIST.train_input)))
        random.shuffle(idx)
        split = int(np.floor(0.2 * len(PairSetMNIST.train_input)))
        train_idx = idx[split:]
        valid_idx = idx[:split]
            
        self.train_idx = train_idx
        self.valid_idx = valid_idx
            
    def __getitem__(self,index) : 
        
        return self.train_input[index], self.train_target[index], self.train_classes[index]
    
    def __len__(self):

        return self.len

########################################################################################################################
            
class Training_set_split(Dataset) :
    """
    Training set splitted  from Training set
    """
    
    def __init__(self,Training_set,rotate,translate,swap_channel) :
        
        train_input =  Training_set.train_input[Training_set.train_idx]
        train_target  = Training_set.train_target[Training_set.train_idx]
        train_classes = Training_set.train_classes[Training_set.train_idx]
        
        #Data augmentation
        if (rotate == True) :
            # find indices of 6 and 9 digits not to rotate them
            indices_9 = (train_classes == 9)
            index_9 = indices_9.nonzero()
            indices_6 = (train_classes == 6)
            index_6 = indices_6.nonzero()
            
            # rotate the images
            empty_images90= torch.rot90(train_input,1,[2,3])
            empty_images180= torch.rot90(train_input,2,[2,3])
            empty_images270= torch.rot90(train_input,3,[2,3])
            
            # replace the rotated 6 and 9 by the original 
            empty_images90[index_9[:,0],index_9[:,1]] = train_input[index_9[:,0],index_9[:,1]]
            empty_images90[index_6[:,0],index_6[:,1]] = train_input[index_6[:,0],index_6[:,1]]
            empty_images180[index_9[:,0],index_9[:,1]] = train_input[index_9[:,0],index_9[:,1]]
            empty_images180[index_6[:,0],index_6[:,1]] = train_input[index_6[:,0],index_6[:,1]]
            empty_images270[index_9[:,0],index_9[:,1]] = train_input[index_9[:,0],index_9[:,1]]
            empty_images270[index_6[:,0],index_6[:,1]] = train_input[index_6[:,0],index_6[:,1]]
            
            # Concatenate the rotated images to the data
            train_input = torch.cat((train_input,empty_images90,empty_images180,empty_images270),dim=0)
            train_classes = torch.cat((train_classes,train_classes, train_classes,train_classes), dim=0)
            train_target = torch.cat((train_target,train_target,train_target,train_target),dim=0)
              
        if (translate == True) :
            background = train_input[0,0,0,0]
            upward = torch.zeros(1000,2,14,14)
            downward = torch.zeros(1000,2,14,14)
            left = torch.zeros(1000,2,14,14)
            right = torch.zeros(1000,2,14,14)
            #translate images
            upward[:,:,:-1,:] = train_input[:1000,:,1:,:]
            downward[:,:,1:,:] = train_input[:1000,:,:-1,:]
            left[:,:,:,:-1] = train_input[:1000,:,:,1:]
            right[:,:,:,1:] = train_input[:1000,:,:,:-1]
            #add the background value to the boundary
            upward[:,:,13,:] = background
            downward[:,:,0,:] = background
            left[:,:,:,13] = background
            right[:,:,:,0] = background
            
            #concatenate the translated images
            train_input = torch.cat((train_input,upward,downward,left,right),dim=0)
            train_classes = torch.cat((train_classes, train_classes[:1000],train_classes[:1000],train_classes[:1000],train_classes[:1000]), dim=0)
            train_target = torch.cat((train_target,train_target[:1000],train_target[:1000],train_target[:1000],train_target[:1000]),dim=0)
            
        if (swap_channel==True):
            
            train_input = torch.cat((train_input, train_input.flip(1)), dim=0)
            train_classes = torch.cat((train_classes, train_classes.flip(1)), dim=0)
            train_target = torch.cat((train_target, (train_classes.flip(1)[:,0] <= train_classes.flip(1)[:,1]).long()),dim=0)
        
        self.len = train_input.shape[0]
        self.train_input =  train_input
        self.train_target  = train_target
        self.train_classes = train_classes
    
    def __getitem__(self,index) : 
        
        return self.train_input[index], self.train_target[index], self.train_classes[index]
    
    def __len__(self):

        return self.len

###########################################################################################################################
    
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
        
        
    
        