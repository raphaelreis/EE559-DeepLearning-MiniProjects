
# baseline model
class Net2C(nn.Module):

    def __init__(self, nb_hidden, name = None):
        
        super(Net2C, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5) # number of input channels is 2.
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, nb_hidden)
        #self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(nb_hidden, 1) # single output
        self.sigmoid = nn.Sigmoid()  
        
    def forward(self, x):
    
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3, stride = 3))
        x = F.relu(self.fc1(x.view(-1, 256)))
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x) 
        return x # returns a probability 

# inspired from LeNet5 but slightly different and modified for binary classification 

class Netcat(nn.Module):      
    
    def __init__(self, name = None):
        
        super(Netcat, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)   # add final layer for binary classificaton
        self.sigmoid = nn.Sigmoid()
  
        
    def forward(self, x):
   
        x = torch.cat((x[:,0], x[:,1]), dim = 2).unsqueeze(dim = 1)   # concatenate channels into 1 channel (input : 1000 * 1 * 14 * 28)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 1))
        #print(x.size())
        x = F.relu(self.fc1(x.view(-1, 480)))
        x = F.relu(self.fc2(x.view(-1, 120)))
        x = self.fc3(x)
        x = self.sigmoid(x) 
        return x # returns a probability




