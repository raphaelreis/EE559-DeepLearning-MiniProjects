from matplotlib import pyplot as pt
import sys
import numpy as np
sys.path.append('..')

def learning_curve(tr_losses, tr_accuracies, valid_losses, valid_accuracies):

    """ Plots learning curve over training epochs for single model """
    
    fig, ax1 = pt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(tr_losses, 'g', linewidth=2, label = 'Train  ')
    ax1.plot(valid_losses, 'r', linewidth=2, label = 'Validation  ')
    ax2 = ax1.twinx()
    
    ax2.set_ylabel('Accuracy')
    ax2.plot(tr_accuracies, 'g--', linewidth=2)
    ax2.plot(valid_accuracies, 'r--', linewidth=2)
    ax1.legend(loc = 'center right')

    pt.show()