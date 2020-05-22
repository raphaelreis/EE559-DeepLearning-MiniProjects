from matplotlib import pyplot as plt
import sys
import numpy as np
sys.path.append('..')
import matplotlib.patches as mpatch
import torch

# simple learning curve for visualization of training over epochs

def learning_curve(tr_losses, tr_accuracies, valid_losses, valid_accuracies):

    """ 
    
     General : Plots learning curve over training epochs for a single training 
     
     Input :
     
         - tr_losses : tensor of the losses of shape (n_epochs,)
         - tr_accuracies : tensor of the accuracies of shape (n_epochs,)
         - valid_losses : tensor of the losses of shape (n_epochs,)
         - valid_accuracies : tensor of the accuracies of shape (n_epochs,)
         
     Output : Graph of the learning curves (losses and accuracies) for the training and validation set
    
    """
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Normalized loss [loss/data size]')
    a1, = ax1.plot(tr_losses, 'g', linewidth=2, label = 'Train loss')
    a2, = ax1.plot(valid_losses, 'r', linewidth=2, label = 'Validation loss')
    ax2 = ax1.twinx()
    
    ax2.set_ylabel('Accuracy [%]')
    b1, = ax2.plot(tr_accuracies, 'g--', linewidth=2, label = 'Train acc')
    b2, = ax2.plot(valid_accuracies, 'r--', linewidth=2, label = 'Validation acc')
    
    t = [a1, a2, b1, b2]
    ax1.legend(t, [t_.get_label() for t_ in t], loc = 'best', fontsize='small')
    

    plt.show()
    
# Boxplot

def boxplot(data,title,models,save = False):

    """ 

     General : Boxplot of the train,validation and test accuracies for each model in models
    
    INPUT :

        - data : a tensor (nb_models,3,10)
        - title : title of the boxplot
        - models : names of the models tested
        - save : boolean to save the figure

    Output : Boxplot. If only one model  mean and standard deviation are printed

    """

    boxdict1, boxdict2,boxdict3 = dict(linestyle='-', linewidth=2, color='black'), dict(linestyle='-', linewidth=2,color='blue'),dict(linestyle='-', linewidth=2, color='gray')
    whiskerdict1, whiskerdict2 ,whiskerdict3 = dict(linestyle='-', linewidth=2, color='black'), dict(linestyle='-', linewidth=2, color='blue'),dict(linestyle='-', linewidth=2, color='gray')
    mediandict = dict(linestyle='--', linewidth=1.5, color='red')

    fig1, ax1 = plt.subplots(1,1,figsize=(12,8))

    ax1.set_title(title)
    
    np.random.seed(0)
    p = 0 
    print(data.shape)
    for i in range (data.shape[0]) :
      ax1.boxplot(data[i,0].view(1,-1), patch_artist=False,positions = [p], widths = 0.3, showfliers=True, showcaps=False, boxprops=boxdict1, whiskerprops=whiskerdict1, 
                      medianprops=mediandict)
      ax1.boxplot(data[i,1].view(1,-1), patch_artist=False,positions = [p+1], widths = 0.3, showfliers=True, showcaps=False, boxprops=boxdict2, whiskerprops=whiskerdict2, 
                      medianprops=mediandict)
      ax1.boxplot(data[i,2].view(1,-1), patch_artist=False,positions = [p+2], widths = 0.3, showfliers=True, showcaps=False, boxprops=boxdict3, whiskerprops=whiskerdict3, 
                      medianprops=mediandict)
      ax1.scatter(np.random.normal(p, 0.05, data[i,0].shape[0]), data[i,0], c='red')
      ax1.scatter(np.random.normal(p+1, 0.05, data[i,1].shape[0]), data[i,1], c='red')
      ax1.scatter(np.random.normal(p+2, 0.05, data[i,2].shape[0]), data[i,2], c='red')
      p +=3
    
    ax1.set_xticks([k+1.0 for k in range(0,p,3)])
    ax1.set_xticklabels(labels =models) 
    ax1.yaxis.grid(True)
    ax1.set_ylabel('Accuracy (%)')

    labels = ['Train', 'Validation', 'Test']
    handles = [mpatch.Patch(facecolor='black'), mpatch.Patch(facecolor='blue'),mpatch.Patch(facecolor='gray') ]
    ax1.legend(handles,labels, loc='best')

    if (save == True) :
      fig1.savefig('Figures/Best_performance.png')

    plt.show()
    
    if (len(models) == 1) :
        
        # mean acc
        mean_acc_train = torch.mean(data[0,0])
        mean_acc_validation = torch.mean(data[0,1])
        mean_acc_test = torch.mean(data[0,2])
        # std acc
        std_acc_train = torch.std(data[0,0])
        std_acc_validation = torch.std(data[0,1])
        std_acc_test = torch.std(data[0,2])

        print('Training |  Mean accuracy: {:.3f} | Standard deviation: {:.3f}\n'.format( mean_acc_train, std_acc_train))
        print('Validation |  Mean accuracy: {:.3f} | Standard deviation: {:.3f}\n'.format( mean_acc_validation, std_acc_validation))
        print('Test |  Mean accuracy: {:.3f} | Standard deviation: {:.3f}\n'.format( mean_acc_test, std_acc_test))
