from models.Nets import *
from models.Basic import *
from models.Inception_Net import *
from models.Le_Net import *
from utils.dlc_practical_prologue import *
from utils.Evaluate import *
from utils.grid_search import *
from utils.loader import *
from utils.metrics import *
from utils.plot import *
from utils.training import *
import argparse


if __name__ == "__main__":
    
    Nets_default = Nets()
    seeds = [1,2,3,4,5,6,7,8,9,10]
    
    train_results, test_losses,test_accuracies = evaluate_model(Nets_default.LeNet_sharing_aux, seeds , plot =False, 
                                                               rotate = True,translate=True,swap_channel = True)
