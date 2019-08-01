import mlp
import mnist
import activations as ac

import numpy as np

import time

np.random.seed(11785)

#initialize neural parameters
learning_rate = 0.004
momentum = 0.996 #0.956
num_bn_layers= 1
mini_batch_size = 10
epochs = 40


# initialize training, validation and testing data
train, val, test = mnist.load_mnist()

net = mlp.MLP(784, 10, [64, 32], [ac.Sigmoid(), ac.Sigmoid(), ac.Sigmoid()], ac.SoftmaxCrossEntropy(), learning_rate,
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
net.save(str(accuracy) + "_acc_nn_model.pkl")