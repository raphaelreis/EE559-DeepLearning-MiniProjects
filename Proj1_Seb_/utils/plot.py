from matplotlib import pyplot as plt
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

def boxplot(data, title='Boxplot'):

    """ 
    plots a single boxplot for a single model type
    
    INPUT : a 2 by (number of seeds tensor) with in each line the model accuracies

    """

    labels = ['Train','Validation', 'Test']

    boxdict1, boxdict2,boxdict3 = dict(linestyle='-', linewidth=2, color='black'), dict(linestyle='-', linewidth=2,color='black'),dict(linestyle='-', linewidth=2, color='black')
    whiskerdict1, whiskerdict2 ,whiskerdict3 = dict(linestyle='-', linewidth=2, color='black'), dict(linestyle='-', linewidth=2, color='black'),dict(linestyle='-', linewidth=2, color='black')
    mediandict = dict(linestyle='--', linewidth=1.5, color='red')

    fig1, ax1 = plt.subplots(1,1,figsize=(10,7))

    ax1.set_title(title)

    bplot = ax1.boxplot(data, patch_artist=False, widths = 0.2, showfliers=True, showcaps=False, boxprops=boxdict1, whiskerprops=whiskerdict1, medianprops=mediandict, labels=labels)
    
    np.random.seed(0)
    
    for i in range(3):
        y = data[i]
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax1.plot(x, y, 'r.', alpha=1)
        

    ax1.yaxis.grid(True)
    ax1.set_xlabel('')
    ax1.set_ylabel('Accuracy (%)')
    
    # mean acc
    mean_acc_train = torch.mean(data[0])
    mean_acc_validation = torch.mean(data[1])
    mean_acc_test = torch.mean(data[2])
    # std acc
    std_acc_train = torch.std(data[0])
    std_acc_validation = torch.std(data[1])
    std_acc_test = torch.std(data[2])
    
    print('Training |  Mean accuracy: {:.3f} | Standard deviation: {:.3f}%\n'.format( mean_acc_train, std_acc_train))
    print('Validation |  Mean accuracy: {:.3f} | Standard deviation: {:.3f}%\n'.format( mean_acc_validation, std_acc_validation))
    print('Test |  Mean accuracy: {:.3f} | Standard deviation: {:.3f}%\n'.format( mean_acc_test, std_acc_test))

