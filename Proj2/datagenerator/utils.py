
from torch import empty

############################ Utils ############################# noqa: E266


def one_hot(y, dims=2):
    y_hot = y.long().view(-1, 1).clone()
    y_one_hot = empty(y_hot.size()[0], dims).\
        fill_(0.).scatter_(1, y_hot, 1)
    return y_one_hot
