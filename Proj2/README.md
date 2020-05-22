# Mini deep-learning framework

The objective of this project is to design a mini *deep learning framework* using only pytorch's
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.


Please check our report for this project here: [Porject2 report](https://github.com/raphaelreis/EE559-DeepLearning-MiniProjects/blob/master/Proj2/EE559_proj2_report.pdf)

## Getting Started

The project is splitted in five folders:
* analysis:
> Notebook analysis and figures
* data generator
> The class to generate the data
* neuralnetworks:
1. base.py: contains the parent class for all modules
2. dropout.py
3. feedforward.py
4. functions.py: contains all the activation functions and loss functions
5. sequential.py: contains the high level learning protocol for the deep network to solve the classification task
* optimizer
> Sochastic gradient descent optimizer
* test
> unit testing

### Prerequisites

The project mainly rely on Pytorch. However, to run the notebook and get the figures it is necessary to install matplotlib.pyplot

## Running the test.py file
**!!for the TAs!!**

1) `python test.py`: run a model with 3 layers and tanh activation function
2) `python test.py -model 2`: run a model with 3 layers and ReLU activation function

Both models have Tanh final activation function

## Running the tests

To run the test go to the proj2 folder and run:
* `python -m unittest -f tests.test_sequential`
* `python -m unittest -f tests.test_feedforward`

## Authors

* **Raphaël Reis Nunes** - *Initial work* - [GitHub](https://github.com/raphaelreis)
* **Vincent Bernaert**
* **Sébastien Emery**


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
