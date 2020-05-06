from matplotlib import pyplot as pt
import sys
import numpy as np
sys.path.append('..')

# simple learning curve for visualization of training over epochs

def learning_curve(tr_losses, tr_accuracies, valid_losses, valid_accuracies):

    """ Plots learning curve over training epochs for single model """
    
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
    
# to do boxplot