# EE559-DeepLearning-MiniProjects

## Project 1 : Comparison of two visible digits in a two channels image
The Goal of this project is to implement different neural network structures to predict if a handwritten digit in the first channel is lesser or equal to the digit in the second channel and 
in particular to test the impact of weight sharing and auxiliary losses.

### Main 

### Data
The data is taken from the MNIST dataset from Yann Lecun website. The lecturer of the EEE-559 lecture at epfl (Fleuret François) provides a python file (dlc_prologue.py) which generates 
our dataset. The dataset is structured as folows :

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

### Performances

### Contributors

Bernaert Vincent, Emery Sébastien and Reis Raphael 