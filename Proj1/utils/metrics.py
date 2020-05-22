




def accuracy(y_pred, target):
    """ return the accuracy of a prediction given the target """

    assert y_pred.shape[0] == len(target), "y_pred and target should be the same shape"

    size = target.shape[0]
    return ((y_pred.argmax(1) == target).sum().float() / float(size)).item()