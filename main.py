import mlp
import mnist
import activations as ac

import numpy as np

import time

np.random.seed(11785)

#initialize neural parameters
learning_rate = 0.008
momentum = 0.856
num_bn_layers= 0
mini_batch_size = 16
epochs = 2


def weight_init(x, y):
    return np.random.randn(x, y)


def bias_init(x):
    return np.zeros((1, x))

# initialize training, validation and testing data
train, val, test = mnist.load_mnist()

net = mlp.MLP(784, 10, [64, 32], [ac.Sigmoid(), ac.Sigmoid(), ac.Sigmoid()], weight_init, bias_init, ac.SoftmaxCrossEntropy(), learning_rate,
          momentum, num_bn_layers)


start = time.time()

#training neural network
net.fit(train, val, epochs, mini_batch_size)
end = time.time()

print("Training time(sec.) =", end-start)

#testing neural network
accuracy = net.validate(test) * 100.0
print("Test Accuracy: " + str(accuracy) + "%")

#save the model
net.save()