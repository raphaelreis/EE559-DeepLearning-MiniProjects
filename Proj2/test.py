import sys
import argparse
from collections import defaultdict


from optimizer.sgd import SGD
# from neuralnetworks.dropout import Dropout
from neuralnetworks.sequential import Sequential
from neuralnetworks.feedforward import Feedforward
from datagenerator.datagenerator import DataGenerator
from neuralnetworks.functions import ReLU, MSE, Tanh


sys.path.insert(0, "../")


def training(mlp, optimizer, loss, epochs):
    # Statistics lists
    loss_history_train = []
    loss_history_test = []
    accuracy_history_train = []
    accuracy_history_test = []
    gradient_checker = defaultdict(list)

    for e in range(epochs):
        # Counters
        correct_train = 0
        correct_test = 0
        loss_stack_train = 0.
        loss_stack_test = 0.

        # Training
        for i, (val, tar) in enumerate(zip(X_train, y_train)):
            optimizer.zero_grad(mlp)
            output = mlp.forward(val)
            loss(output, tar)
            if output.abs().argmax() == tar.argmax():
                correct_train += 1
            loss_stack_train += loss.value.item()
            mlp.backward(loss)
            optimizer.step(mlp)

        # Gradient Checker:
        for i, layer in enumerate(mlp.param()):
            if layer != []:
                gradient_checker["layer{}_w".format(i)]\
                    .append(layer[1].mean().item())
                gradient_checker["layer{}_b".format(i)]\
                    .append(layer[3].mean().item())

        # Testing
        for val, tar in zip(X_test, y_test):
            output = mlp.forward(val)
            loss(output, tar)
            if output.abs().argmax() == tar.argmax():
                correct_test += 1
            loss_stack_test += loss.value.item()

        # Metrics evaluation and printing
        l_train = loss_stack_train / X_train.shape[0]
        l_test = loss_stack_test / X_test.shape[0]
        acc_train = correct_train / X_train.shape[0]
        acc_test = correct_test / X_test.shape[0]

        accuracy_history_train.append(acc_train)
        accuracy_history_test.append(acc_test)
        loss_history_train.append(l_train)
        loss_history_test.append(l_test)

        print("epoch: ", e, "| train_loss: ", l_train, " | train_acc: ",
              acc_train, " | test_loss: ", l_test, " | test_acc: ", acc_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training models for cercle detection')
    parser.add_argument('-model', type=int, help='type 1 or 2', default=1)
    parser.add_argument('-epochs', type=int, default=40)
    args = parser.parse_args()

    # Generate the data
    dg = DataGenerator(1000)
    X_train, y_train, X_test, y_test = dg.get_data()

    mlp = Sequential()
    optimizer = None
    loss = None
    if args.model == 1:
        # Model 1
        # 3 hidden layers, tanh activation function, lr=0.01, SGD, MSE
        mlp.add(Feedforward(2, 25))
        mlp.add(Tanh())
        mlp.add(Feedforward(25, 25))
        mlp.add(Tanh())
        mlp.add(Feedforward(25, 25))
        mlp.add(Tanh())
        mlp.add(Feedforward(25, 2))
        mlp.add(Tanh())
        lr = 0.01
        optimizer = SGD(lr)
        loss = MSE()
    elif args.model == 2:
        # Model 2
        # 3 hidden layers, relu activation function, lr=0.001, SGD, MSE
        mlp.add(Feedforward(2, 25))
        mlp.add(ReLU())
        mlp.add(Feedforward(25, 25))
        mlp.add(ReLU())
        mlp.add(Feedforward(25, 25))
        mlp.add(ReLU())
        mlp.add(Feedforward(25, 2))
        mlp.add(Tanh())
        lr = 0.01
        optimizer = SGD(lr)

        loss = MSE()

    training(mlp, optimizer, loss, args.epochs)
