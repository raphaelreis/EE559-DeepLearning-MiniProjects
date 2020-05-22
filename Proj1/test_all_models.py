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
    
    # Construct class with dictionnaries of network models as attirbute
    Nets_default = Nets()
    # Set the seeds
    seeds = [1,2,3,4,5,6,7,8,9,10]
    
    # Call the training/ validation and testing procedure 10 time dor the seeds above
    train_results_basic,test_losses_basic, test_accuracies_basic = evaluate_model(Nets_default.Net2c,seeds,plot = False,statistics = False)
    train_results_share,test_losses_share, test_accuracies_share = evaluate_model(Nets_default.LeNet_sharing,seeds,plot = False,
                                                                                  statistics = False)
    train_results_aux,test_losses_aux, test_accuracies_aux = evaluate_model(Nets_default.LeNet_sharing_aux,seeds,plot = False,
                                                                            statistics = False)
    train_results_google,test_losses_google, test_accuracies_google = evaluate_model(Nets_default.Google_Net,seeds,plot = False, 
                                                                                     statistics = False)
    train_results_aux_augm, test_losses_aux_augm ,test_accuracies_aux_augm = evaluate_model(Nets_default.LeNet_sharing_aux, seeds ,
                                                                                            plot =False, statistics = False, rotate = True,
                                                                                            translate=True,swap_channel = True)
    # reshape the data for boxplot
    data_basic = torch.stack([train_results_basic[:,1,39], train_results_basic[:,3,39] ,test_accuracies_basic])
    data_share = torch.stack([train_results_share[:,1,39], train_results_share[:,3,39] , test_accuracies_share])
    data_google = torch.stack([train_results_google[:,1,39], train_results_google[:,3,39] , test_accuracies_google])
    data_aux = torch.stack([train_results_aux[:,1,39], train_results_aux[:,3,39] , test_accuracies_aux])
    data_aux_augm = torch.stack([train_results_aux_augm[:,1,39], train_results_aux_augm[:,3,39] , test_accuracies_aux_augm])

    data = torch.cat([data_basic.view(1,3,10),data_share.view(1,3,10),data_google.view(1,3,10),data_aux.view(1,3,10),
                      data_aux_augm.view(1,3,10)], dim = 0)
    
    # Boxplot
    Title =  "Models accuracies"
    models_name = ['Net2c','Lenet_sharing','Google Net','Lenet_sharing_aux','Lenet_sharing_aux_augm' ]

    boxplot(data, Title, models_name,True,'Figures/All_models.png')