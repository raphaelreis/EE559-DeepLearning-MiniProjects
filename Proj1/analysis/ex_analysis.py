import torch 
from matplotlib import pyplot
import sys
import numpy as np
sys.path.append('..')
from torch import nn 
from torch.nn import functional as F
from torch import optim
from utils.loader import load
from utils.loader import PairSetMNIST
import torch.utils.data as dt
from torch.utils.data import Dataset, DataLoader
from utils.plot import learning_curve
from utils.metrics import accuracy, compute_nb_errors

# model architectures

class LeNet_aux_sequential(nn.Module):
    
    """    
    Weight sharing + Auxiliary loss
    
    """
    def __init__(self):
        super(LeNet_aux_sequential, self).__init__()
        # convolutional weights for digit reocgnition shared for each image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # weights for binary classification 
        self.fc3 = nn.Linear(20, 60)
        self.fc4 = nn.Linear(60, 90)
        self.fc5 = nn.Linear(90, 2)
        
    def forward(self, input_):    
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        
        # forward pass for the second image 
        y = F.relu(F.max_pool2d(self.conv1(y), kernel_size=2, stride=2))
        y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
        y = F.relu(self.fc1(y.view(-1, 256)))
        y = self.fc2(y)
        
        # concatenate layers  
        z = torch.cat([x, y], 1)
        
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        
        return x, y, z
    
    
# train function

def train_aux (model, train_data, validation_data, mini_batch_size=100, optimizer = optim.SGD,
                criterion = nn.CrossEntropyLoss(), n_epochs=50, eta=1e-1, lambda_l2 = 0, alpha=0.5, beta=0.5):
    
    """
    Train network with auxiliary loss + weight sharing and record train/validation history
    
    """
    train_acc = []
    train_losses = []
    valid_acc = []
    valid_losses = []
    
    model.train()
    optimizer = optimizer(model.parameters(), lr = eta)
    
    for e in range(n_epochs):
        epoch_loss = 0
        train_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
        
        for i, data in enumerate(train_loader, 0):
            
            input_, target_, classes_ = data
            class_1, class_2, out = model(input_)
            aux_loss1 = criterion(class_1, classes_[:,0])
            aux_loss2 = criterion(class_2, classes_[:,1])
            out_loss  = criterion(out, target_)
            net_loss = (alpha * (out_loss) + beta * (aux_loss1 + aux_loss2) )
            epoch_loss += net_loss
            
            if lambda_l2 != 0:
                for p in model.parameters():
                    epoch_loss += lambda_l2 * p.pow(2).sum() # add an l2 penalty term to the loss 
            
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            
        tr_loss, tr_acc = compute_metrics(model, train_data)
        val_loss, val_acc = compute_metrics(model, validation_data)
        
        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        valid_acc.append(val_acc)
        valid_losses.append(val_loss)
            
        print('Train Epoch: {}  | Loss {:.6f}'.format(
                e, epoch_loss.item()))
        
    return train_losses, train_acc, valid_losses, valid_acc


# compute loss and accuracy given a model and test (or validation) data

def compute_metrics(model, test_data, mini_batch_size=100, criterion = nn.CrossEntropyLoss()):
    
    """
    function to calculate prediction accuracy + loss of a cnn with auxiliary loss
    
    """
    test_loader = DataLoader(test_data, batch_size=mini_batch_size, shuffle=False)
    model.eval()
    test_loss = 0
    nb_errors = 0
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader, 0):
            input_, target_, classes_ = data
            _, _, output = model(input_) 
            batch_loss = criterion(output, target_)
            test_loss += batch_loss     
            nb_errors += compute_nb_errors(output, target_)
            
        acc = 100*(1 - (nb_errors/test_data.len) )
        test_loss = test_loss/test_data.len     # normalize loss
              
        return test_loss.item(), acc
    
    
    
# simple validation 


def validate_model(net_type, mini_batch_size=100, optimizer = optim.SGD,
                 criterion = nn.CrossEntropyLoss(), n_epochs=40, eta=1e-1, 
                 lambda_l2 = 0, alpha=0.5, beta=0.5, plot=True): 

    """ Training / validation over n_epochs + testing a full test set"""
    
    train_data = PairSetMNIST(train=True)
    valid_data = PairSetMNIST(valid=True)
    test_data  = PairSetMNIST(test=True)
    
    model = net_type()
    train_losses, train_acc, valid_losses, valid_acc = train_aux(model, train_data, valid_data, mini_batch_size, optimizer, criterion, n_epochs, eta, lambda_l2, alpha, beta)
    
    if plot:
        
        learning_curve(train_losses, train_acc, valid_losses, valid_acc)

    test_loss, test_accuracy = compute_metrics(model, test_data)
    
    print('\nTest Set | Loss: {:.4f} | Accuracy: {:.2f}%\n'.format(test_loss, test_accuracy))
    
# to do : cross validation


# evaluation and final prediction statistics on large test set

def evaluate_model(net_type, n_trials=10, mini_batch_size=100, optimizer = optim.SGD,
                 criterion = nn.CrossEntropyLoss(), n_epochs=40, eta=1e-1, 
                 lambda_l2 = 0, alpha=0.5, beta=0.5, plot=True): 
    
    """ 10 rounds of training / validation + testing metrics statistics  """
    
    train_results = torch.empty(n_trials, 4, n_epochs)
    test_losses = []
    test_accuracies = []
    
    for n in range(n_trials):
    
        train_data = PairSetMNIST(train=True)
        valid_data = PairSetMNIST(valid=True)
        test_data  = PairSetMNIST(test=True)
        
        model = net_type()

        train_losses, train_acc, valid_losses, valid_acc = train_aux(model, train_data, valid_data, mini_batch_size, optimizer, criterion, n_epochs, eta, lambda_l2, alpha, beta)
        train_results[n,] = torch.tensor([train_losses, train_acc, valid_losses, valid_acc])
        test_loss, test_acc = compute_metrics(model, test_data)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if plot:
            
            learning_curve(train_losses, train_acc, valid_losses, valid_acc)
            
            
        
        print('\nTrial {:d} | Test Loss: {:.4f} | Test Accuracy: {:.2f}%\n'.format(n, test_loss, test_acc))
        
    return train_results, test_losses, test_accuracies
        
