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
    Train network with auxiliary loss + weight sharing and record train/validation history
    
    """
    train_acc = []
    train_losses = []
    valid_acc = []
    valid_losses = []
    
    optimizer = optimizer(model.parameters(), lr = eta, weight_decay = lambda_l2)
    
    train_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
    
    for e in range(n_epochs):
        epoch_loss = 0
        model.train(True)
        for i, data in enumerate(train_loader, 0):
            
            input_, target_, classes_ = data

            input_ = input_.to(device)
            target_ = target_.to(device)
            classes_ = classes_.to(device)
            
            if (model.__class__.__name__ == 'LeNet_sharing_aux' or  model.__class__.__name__ == 'Google_Net') :
                class_1, class_2, out = model(input_)
                aux_loss1 = criterion(class_1, classes_[:,0])
                aux_loss2 = criterion(class_2, classes_[:,1])
                out_loss  = criterion(out, target_)
                net_loss = (alpha * (out_loss) + beta * (aux_loss1 + aux_loss2) ) 
            else :
                out = model(input_)
                net_loss  = criterion(out, target_)

            epoch_loss += net_loss
            
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            
        tr_loss, tr_acc = compute_metrics(model, train_data, device)
        val_loss, val_acc = compute_metrics(model, validation_data, device)
        
        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        valid_acc.append(val_acc)
        valid_losses.append(val_loss)
            
        #print('Train Epoch: {}  | Loss {:.6f}'.format(
         #       e, epoch_loss.item()))
        
    return train_losses, train_acc, valid_losses, valid_acc