import torch
import numpy as np
import random
import utils.dlc_practical_prologue as prologue
from torch.utils.data import Dataset, DataLoader



def load():
    '''Load the data in the format required by the project from the prologue file given
    
        Returns : tuple
        
        tuple[0]: train
        tuple[1]:target
        tuple[2]: classes
    '''
    return prologue.generate_pair_sets(1000)

##########################################################################################################################################

class PairSetMNIST(Dataset):
    
    """
     A class that inherite from Dataset of pytorch to automatically handle batches and shuffling of the data when passed to a dataloader
     
     Load the train and test data from load()
     
     Initialize the classes attribut with the train and test data :
      
         - train_input : 1000x2x14x14 (two-channels images)
         - train_target : 1000x1 (target -> digit in the first channel lesser or equal to the one in the second channel)
         - train_classes :  1000x2 contain the label of each digit in the pair of digit
         
         - test_input : 1000x2x14x14 (two-channels images)
         - test_target : 1000x1 (target -> digit in the first channel lesser or equal to the one in the second channel)
         - test_classes :  1000x2 contain the label of each digit in the pair of digit
     
     => This class will be needed as an input of the other classes to separate this dataset in specific datasets : train,validation and 
        test on which we can call a data loader from pytorch
    """
    
    # constructor
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
       
    # iterator             
    def __getitem__(self, index ):
        
        return self.train_input[index], self.train_target[index], self.train_classes[index], self.test_input[index],                       self.test_target[index],self.test_classes[index]

#################################################################################################################################

class Test_set(Dataset) :
    """
     A class that inherite from Dataset of pytorch to automatically handle batches and shuffling of the data when passed to a dataloader
     
     Input : PairSetMNIST 
     
     Only keep the test set in the attribut of the class defined as in PairSetMNIST
    """
    # constructor
    def __init__(self,PairSetMNIST) :
        
        self.len = PairSetMNIST.test_input.shape[0]
        self.test_input  = PairSetMNIST.test_input
        self.test_target  = PairSetMNIST.test_target
        self.test_classes = PairSetMNIST.test_classes
       
    # iterator          
    def __getitem__(self, index):
        
        return self.test_input[index], self.test_target[index], self.test_classes[index]
    
    
    def __len__(self):

        return self.len

#################################################################################################################################
    
class Training_set (Dataset) :
    """
    A class that inherite from Dataset of pytorch to automatically handle batches and shuffling of the data when passed to a dataloader
     
    Input : PairSetMNIST 
     
    Only keep the train set in the attribut of the class defined as in PairSetMNIST
    
    Randomly split the indices of the train_input tensor :
        
        - train_idx -> 0.8 of the training pair
        - valid_idx -> 0.2 of the training pair
        
    => This class will be needed to split the train in a train and valid dataset by using the list of indices split in this class
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
    A class that inherite from Dataset of pytorch to automatically handle batches and shuffling of the data when passed to a dataloader
     
    Input : Training_set -> set as training the indices of training from Training_set
    
    Parameters :
        
        - rotate : if true rotate each digit in the pair of digit by 90, 180,270 degrees except 6 and 9 to avoid confusion
        - translate : if true translate the original digit in four direction 
                       -> by one pixel upward
                       -> by one pixel downward
                       -> by one pixel to the right
                       -> by one pixel to the left
        - swap_channels : if True swap the channels 
        
        => data augmentation on the training set
    """
    
    # Constructor
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
            # get the value of the background pixel to set the pixel value  at the boundary in the opposite of the direction of translation
            background = train_input[0,0,0,0]
            # initialize tensors where to put the translated images
            upward = torch.zeros(800,2,14,14)
            downward = torch.zeros(800,2,14,14)
            left = torch.zeros(800,2,14,14)
            right = torch.zeros(800,2,14,14)
            #translate images
            upward[:,:,:-1,:] = train_input[:800,:,1:,:]
            downward[:,:,1:,:] = train_input[:800,:,:-1,:]
            left[:,:,:,:-1] = train_input[:800,:,:,1:]
            right[:,:,:,1:] = train_input[:800,:,:,:-1]
            #add the background value to the boundary
            upward[:,:,13,:] = background
            downward[:,:,0,:] = background
            left[:,:,:,13] = background
            right[:,:,:,0] = background
            
            #concatenate the translated images
            train_input = torch.cat((train_input,upward,downward,left,right),dim=0)
            train_classes = torch.cat((train_classes, train_classes[:800],train_classes[:800],train_classes[:800],train_classes[:800]), dim=0)
            train_target = torch.cat((train_target,train_target[:800],train_target[:800],train_target[:800],train_target[:800]),dim=0)
            
        if (swap_channel==True):
            
            # swap the channels by using the flip function from pytorch
            train_input = torch.cat((train_input, train_input.flip(1)), dim=0)
            train_classes = torch.cat((train_classes, train_classes.flip(1)), dim=0)
            train_target = torch.cat((train_target, (train_classes.flip(1)[:,0] <= train_classes.flip(1)[:,1]).long()),dim=0)
        
        # training 
        self.len = train_input.shape[0]
        self.train_input =  train_input
        self.train_target  = train_target
        self.train_classes = train_classes
    
    # iterator
    def __getitem__(self,index) : 
        
        return self.train_input[index], self.train_target[index], self.train_classes[index]
    
    def __len__(self):

        return self.len

###########################################################################################################################
    
class Validation_set(Dataset) :
    """
    A class that inherite from Dataset of pytorch to automatically handle batches and shuffling of the data when passed to a dataloader
     
    Input : Training_set -> set as validation the indices of validation from Training_set 
    """
    
    # Constructor
    def __init__(self,Training_set) :
        
        self.len = len(Training_set.valid_idx)
        self.valid_input =  Training_set.train_input[Training_set.valid_idx]
        self.valid_target  = Training_set.train_target[Training_set.valid_idx]
        self.valid_classes = Training_set.train_classes[Training_set.valid_idx]
    
    # iterator 
    def __getitem__(self,index) : 
        
        return self.valid_input[index], self.valid_target[index], self.valid_classes[index]
    
    def __len__(self):

        return self.len
        
        
    
        