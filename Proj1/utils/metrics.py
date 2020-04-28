
def accuracy(y_pred, target):
    """ return the accuracy of a prediction given the target """

    assert y_pred.shape[0] == len(target), "y_pred and target should be the same shape"

    return (y_pred.argmax(1) == target).sum().float() / float(target.shape[0])

def compute_nb_errors(y_pred, target):
    
    
    """ return the number of errors of a prediction given the target"""
    
    assert y_pred.shape[0] == len(target), "y_pred and target should be the same shape"
    
    return float(len(target) - (y_pred.argmax(1) == target).sum())