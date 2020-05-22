from models import *
from utils import *



if __name__ == "__main__":
    
    # parse args
    args = args_parser(description='EE-559 project 1 executable')
    
    #args.add_argument('--', default=False, help='Use the full set, can take ages (default False)')
    
    Nets_default = Nets()
    seeds = [1,2,3,4,5,6,7,8,9,10]
    
    
    train_results, test_lossestest_accuracies = evaluate_model(Nets_default.LeNet_sharing_aux, seeds , plot =False, 
                                                               rotate = True,translate=True,swap_channel = True)