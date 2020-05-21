# EE559-DeepLearning-MiniProjects

## Project 1 : Comparison of two visible digits in a two channels image
The Goal of this project is to implement different neural network structures to predict if a handwritten digit in the first channel is lesser or equal to the digit in the second channel and 
in particular to test the impact of weight sharing and auxiliary losses.

### Main
The main.py file contain the executable of our project, it can be called as follows :
  

### Data
The data is taken from the MNIST dataset from Yann Lecun website. The lecturer of the EEE-559 lecture at epfl (Fleuret François) provides a python file (dlc_prologue.py) which generates 
our dataset. The dataset is structured as follows :

   Name      | Tensor dimension | Type  |                 Content
-------------|------------------|-------|-------------------------------------
train input  | 1000x2x14x14     |float32|Images                 
train target | 1000             | int64 |Class to predict {0,1}
train classes| 1000x2           | int64 |Classes of the two digits {0,1,...,9}          
test input   | 1000x2x14x14     |float32|Images 
test target  | 1000             |int64  |Class to predict {0,1}
test classes | 1000x2           |int64  |Classes of the two digits {0,1,...,9}


### Models 
This folder contain the networks class  implemented in this project using the pytorch library and a class which  contains a dictionnary of this class (name and parameters).

* Basic.py
	* Class Net2c : A basic CNN without weight sharing and auxiliary losses 
* Le_Net.py
	* Class LeNet_sharing : A Lenet inspired CNN over the two channels with weight sharing followed by FC layers for binary classification.
	* Class Lenet_sharing_aux : A Lenet inspired CNNN over the two channels with weight sharing and auxiliary loss for digit recognition followed by FC layers for binary classification.
* Inception_Net.py 
	* Class Google_Net : A google net inspired CNN which used a classical inception block over the two channels with weight sharing followed by a CNN and auxiliary loss for digit recognition and finally FC layers for binary classification. 
* Nets.py
	* Class Nets : 
		* Attributes : Four dictionaries containing by default the name of the network, the learning rate to use and the parameters to initialize it which were tuned.
		* Functions: Three function to perform grid search on the parameters of Net2c,LeNet_sharing and Lenet_sharing_aux
### Utils
This folder contain all the functions used to train,tune and evaluate a model as well as the data management class.

* dlc_prologue.py : A file provided by the lecturer used to generate the data as explained above.
* loader.py : 
	* load : a function which calls and return the data
	* Class PairSetMNIST : Generate the data calling load and store it in the classes' attributes
		* Class Test_set : recover the test data from PairSetMNIST and store it in the classes' attributes
		* Class Training_set : recover the test data from PairSetMNIST and store it in the classes' attributes. Generate a training and validation set by randomly splitting the indices of the training in training(0.89 and validation (0.2).
			* Class Training_set_split : the final training set 80% recovered from Training_set which can be augmented by rotation and translation of the digits as well as swapping the two channels
			* Class Validation_set : the final training set 20% recovered from Training_set 
* plot.py :
	* learning_curve : plot the training and validation losses and accuracy of a single training
	* boxplot : Boxplot of the training, validation and test accuracies at the end of the training by repeating the procedure for multiple seed
* metrics.py :
	* accuracy : compute the accuracy given a vector of prediction and target
	* compute_nb_errors : compute the number of errors given a vector of prediction and target
	* compute_metrics : function to calculate the prediction accuracy and  the loss of a model on a data
* training.py :
	*train_model : train  an initialized neural network model and record train/validation history
* Evaluate.py :
	* validate_model : Train a neural network model given its dictionnary to initialize it and a seed for initialization. Record the training and validation accuracies and compute the test accuracy.
	* evaluate_model : Repeat a ten times training/validation procedure on given seeds to initialize the model and the data. Record the training and validation accuracies and compute the test accuracy at each seed, then compute statistics (mean and standard deviation).

### Performances

### Contributors

Bernaert Vincent, Emery Sébastien and Reis Raphael 