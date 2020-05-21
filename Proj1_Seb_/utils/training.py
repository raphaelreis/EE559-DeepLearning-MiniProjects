import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
from utils.metrics import compute_metrics
from torch.utils.data import Dataset, DataLoader


# General training function for already initialized model


def train_model(model, train_data, validation_data, device, mini_batch_size=100, optimizer = optim.Adam,
                criterion = nn.CrossEntropyLoss(), n_epochs=40, eta=1e-3,lambda_l2=0, alpha=0.5, beta=0.5):
    
    """
    Train  a neural network model and record train/validation history
    
    Input :
        
        - model : Neural network class to train
        - train_data : data used for training the network : input,target and classes
        - Validation data : unseen data used to evaluate the performance of the network at each epoch on evaluation mode -> input,target 
                            and classes
        - mini_batch_size : batch size in which the data is splitted by the data loader -> default 100
        - optimizer : optimizer used to optimize the network -> default Adam
        -criterion : loss functionto minimize to train the network -> default cross entropy loss
        - n_epochs : number of steps to optimize the network -> default 40
        - eta : learning rate used by the optimizer -> default 10e-3 
        - lambda_l2 : weight penalty term (weight decay) -> default 0
        - alpha : weight term  of the binary loss in the overall loss-> default 0.5
        - beta : weight term  of the auxiliary loss in the overall loss-> default 0.5
    
    Output :
    
        - List of the train accuracy at each epoch
        - List of the train losses at each epoch
        - List of the validation accuracy at each epoch
        - List of the validation loss at each epoch
    
    """
    # Accuracy and loss history of the train and validation data
    train_acc = []
    train_losses = []
    valid_acc = []
    valid_losses = []
    
    # optimizer class initialized with the parameters passed in the constructor
    optimizer = optimizer(model.parameters(), lr = eta, weight_decay = lambda_l2)
    # data loader 
    train_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
    
    for e in range(n_epochs):
        epoch_loss = 0
        # set the model to train mode
        model.train(True)
        for i, data in enumerate(train_loader, 0):
            
            # get the data from the batch
            input_, target_, classes_ = data

            input_ = input_.to(device)
            target_ = target_.to(device)
            classes_ = classes_.to(device)
            
            # check the name of the model to know if the output contain auxiliary loss 
            if (model.__class__.__name__ == 'LeNet_sharing_aux' or  model.__class__.__name__ == 'Google_Net') :
                # get model output
                class_1, class_2, out = model(input_)
                # compute the different losses
                aux_loss1 = criterion(class_1, classes_[:,0])
                aux_loss2 = criterion(class_2, classes_[:,1])
                out_loss  = criterion(out, target_)
                # Overall loss to minimize
                net_loss = (alpha * (out_loss) + beta * (aux_loss1 + aux_loss2) ) 
            else :
                # get the model output
                out = model(input_)
                # Compute the overall loss to minimize
                net_loss  = criterion(out, target_)

            # overral loss on the batch
            epoch_loss += net_loss
            
            # backward 
            optimizer.zero_grad()
            net_loss.backward()
            # gradient step
            optimizer.step()
        
        # compute the loss and accuracy on the whole batch for the training for the epoch
        tr_loss, tr_acc = compute_metrics(model, train_data, device)
        # compute the loss and accuracy on the validation set for the epoch
        val_loss, val_acc = compute_metrics(model, validation_data, device)
        
        # Save the metrics in the list
        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        valid_acc.append(val_acc)
        valid_losses.append(val_loss)
        
    return train_losses, train_acc, valid_losses, valid_acc