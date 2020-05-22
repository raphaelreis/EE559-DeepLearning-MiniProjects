

class Module (object):
    '''Base class of all modules '''
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def param(self):
        return []
