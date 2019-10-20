# Import libraries

from __future__ import print_function

import numpy as np

import sklearn

import pandas as pd

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.ops import resources


# Ignore all GPUs, tf random forest does not benefit from it.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import data

data = pd.read_csv('data1.csv')

#Extract feature and target np arrays (inputs for placeholders)

input_x = data.iloc[:, 0:-1].values

input_y = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size = 0.25, random_state = 0)


data1 = data.iloc[:,:].values

data1

# Parameters
num_steps = 500 # Total steps to train
num_classes = 6 # The 10 digits
num_features = 4 # Each image is 28x28 pixels
num_trees = 12
max_nodes = 10


# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()



# Build the Random Forest

forest_graph = tensor_forest.RandomForestGraphs(hparams)


# Get training graph and loss

train_op = forest_graph.training_graph(X, Y)

loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy

infer_op, _, _ = forest_graph.inference_graph(X)

correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))

accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initialize the variables (i.e. assign their default value) and forest resources

init_vars = tf.group(tf.global_variables_initializer(),
                     resources.initialize_resources(resources.shared_resources()))



# Start TensorFlow session

sess = tf.Session()



# Run the initializer

sess.run(init_vars)

# Training

for i in range(1, num_steps + 1):

    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})

    if i % 50 == 0 or i == 1:

        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})

        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))


print("Done training")


# Test Model

print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))

