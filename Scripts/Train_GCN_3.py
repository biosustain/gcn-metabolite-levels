#!/usr/bin env python

from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import lasagne
import math
# import matplotlib.pyplot as plt
# from IPython import display
import random
import scipy
import json
# import pandas as pd

class GCNLayer(lasagne.layers.Layer):
    """
    A Graph Convolution Net layer.
    
    A_hat: A "square" 3-dimensional tensor, consisting of one or more normalized adjacency matrices, stacked in
        the second dimension. E.g. shape (2, 1000, 1000)
    num_out_features: The number of features to extract from the input.
    num_graphs: The number of different adjacency matrices contained in A_hat. Must be equal to A_hat.shape[0]
    nonlinearity: A theano-compatible nonlinearity function.
    """
    def __init__(
        self, incoming, A_hat, num_out_features=3, num_graphs=3,
        W=lasagne.init.GlorotUniform(),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        **kwargs
    ):
        
        super(GCNLayer, self).__init__(incoming, **kwargs)
        self.A_hat = A_hat
        self.num_graphs = num_graphs
        self.num_out_features = num_out_features
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.linear
        self.nonlinearity = nonlinearity
        
        # print([self.num_graphs, incoming.output_shape[2], num_out_features])
        self.W = self.add_param(W, [self.num_graphs, incoming.output_shape[2], num_out_features], name="W")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_out_features)
    
    def get_output_for(self, input, **kwargs):
        A_X = T.tensordot(self.A_hat, input, axes=[1, 1]).transpose(2, 1, 0, 3)
        Z = T.tensordot(A_X, self.W, axes=2)
        return self.nonlinearity(Z)
    
def special_softmax(t):
    """
    Function that applies the softmax function over the last dimension, keeping the remaining shape
    of the input tensor intact.
    """
    #num = T.exp(t)
    #den = num.sum(2, keepdims=True)
    #return num / den
    t2 = t.reshape((-1, t.shape[-1]))
    t2 = lasagne.nonlinearities.softmax(t2)
    return t2.reshape(t.shape)

class ConfusionMatrix(object):
    def __init__(self, preds, reals):
        assert len(preds) == len(reals)
        self.preds = preds
        self.reals = reals
        classes = sorted(set(preds) | set(reals))
        mat = np.zeros([len(classes), len(classes)], dtype="int")
        for real, pred in zip(reals, preds):
            mat[classes.index(real), classes.index(pred)] += 1
        self.mat = mat     
            
    def __str__(self):
        return str(self.mat)
    
    def bac(self):
        return (self.mat.diagonal() / self.mat.sum(1)).mean()
        

results_obj = []

shuffle = True

for training_round in range(3):

    with open("../Data/Genome-wide/Measured_metabolite_indices.json") as infile:
        data_indices = json.load(infile)
    
    loaded = np.load("../Data/NN_e_coli_data.npz")
    data = dict(zip(loaded["arr_1"], loaded["arr_0"]))

    del loaded


    X = data["X"].astype("float32")
    Y = data["Y"].astype("float32")

    splitpoint = int(X.shape[0] * 0.8)

    order = random.sample(range(len(X)), len(X))
    X = X[order]
    Y = Y[order]

    # Only predict measured metabolites
    data_mask = np.zeros(shape=(Y.shape[0], Y.shape[1], 1), dtype="float32")
    data_mask[:, data_indices] = 1

    Y = np.concatenate([(Y == 0).astype("int"), (Y == 1).astype("int")], axis=2)
    Y = (Y * data_mask).astype("float32")

    A_hat = data["adjacency"].astype("float32").transpose(2, 0, 1)

    # Amplify knockout signal
    X[:, :, 0] *= 3000
    
    if shuffle:
        print("SHUFFLING!")
        np.random.shuffle(X)
    
    X_train = X[:splitpoint]
    Y_train = Y[:splitpoint]
    X_test  = X[splitpoint:]
    Y_test  = Y[splitpoint:]
    


    mask_train = data_mask[:splitpoint]
    mask_test = data_mask[splitpoint:]

    cat_counts = Y.sum(1).sum(0)
    cat_factors = [cat_counts.max() / cnt for cnt in cat_counts]
    # print(cat_counts)
    # print(cat_counts[0] / sum(cat_counts))

    train_cat_counts = Y_train.sum(1).sum(0)
    train_cat_factors = [train_cat_counts.max() / cnt for cnt in train_cat_counts]
    # print(train_cat_counts)
    # print(train_cat_counts[0] / sum(train_cat_counts))
    # 
    # print(np.array(cat_counts) - np.array(train_cat_counts))

    for i, fact in enumerate(train_cat_factors):
        mask_train[:, :, 0] *= (Y_train[:, :, i] == 1).astype("int") * (fact - 1) + 1
    


    BATCH_SIZE = 100
    NUM_INPUTS = X_train.shape[2]
    NUM_OUTPUTS = Y.shape[-1]

    x_sym = T.tensor3("x_sym")
    y_sym = T.tensor3("y_sym")
    A_sym = T.tensor3("A_sym")
    ymask_sym = T.tensor3("ymask_sym")

    l_in = lasagne.layers.InputLayer((None, X.shape[1], NUM_INPUTS))

    l_1 = GCNLayer(l_in, A_sym, num_out_features=10, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 5
    l_2 = GCNLayer(l_1, A_sym, num_out_features=10, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 7
    l_3 = GCNLayer(l_2, A_sym, num_out_features=10, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 9
    l_4 = GCNLayer(l_3, A_sym, num_out_features=10, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 13
    # l_5 = GCNLayer(l_4, A_sym, num_out_features=7, nonlinearity=lasagne.nonlinearities.leaky_rectify)

    l_concat = lasagne.layers.ConcatLayer([
        l_1,
        l_2,
        l_3,
        l_4,
        # l_5
    ], axis=2)

    # Dense: 50 units, leaky_rectify
    # l_slice = lasagne.layers.SliceLayer(l_concat, indices=slice(0, 310), axis=1)
    l_dense = lasagne.layers.DenseLayer(l_concat, num_units=20, num_leading_axes=2, nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_drop = lasagne.layers.DropoutLayer(l_dense)

    l_out = lasagne.layers.DenseLayer(l_drop, num_units=NUM_OUTPUTS, num_leading_axes=2, nonlinearity=special_softmax)


    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    print(sum(w.eval().size for w in all_params), "parameters")

    print("Input layer", lasagne.layers.get_output(l_in, inputs={l_in: x_sym}).eval({x_sym: X[:100]}).shape)

    print("Conv layer", lasagne.layers.get_output(l_1, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)

    print("Conv layer", lasagne.layers.get_output(l_2, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)

    print("Concat layer", lasagne.layers.get_output(l_concat, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)

    print("Dense layer", lasagne.layers.get_output(l_dense, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)

    print("Drop layer", lasagne.layers.get_output(l_drop, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)

    print("Output layer", lasagne.layers.get_output(l_out, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)


    # Retrieve network output
    train_out = lasagne.layers.get_output(l_out, inputs={l_in: x_sym}, deterministic=False)
    eval_out = lasagne.layers.get_output(l_out, inputs={l_in: x_sym}, deterministic=True)

    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # cost = T.nnet.categorical_crossentropy(train_out+1e-8, y_sym).mean()
    # cost = lasagne.objectives.squared_error(train_out, y_sym)
    cost = lasagne.objectives.categorical_crossentropy(train_out+1e-8, y_sym)
    cost = lasagne.objectives.aggregate(cost, weights=ymask_sym[:, :, 0], mode="mean")


    all_grads = T.grad(cost, all_params)

    updates = lasagne.updates.adamax(all_grads, all_params, learning_rate=0.06) #, learning_rate=0.002)

    f_eval = theano.function([x_sym, A_sym],
                         eval_out, on_unused_input='warn')

    cost_eval = theano.function([x_sym, y_sym, A_sym, ymask_sym],
                         cost, on_unused_input='warn')

    f_train = theano.function(
        [x_sym, y_sym, A_sym, ymask_sym],
        cost,
        updates=updates, on_unused_input='warn'
    )


    # Do the training
    EPOCHS = 7
    BATCH_SIZE = 100

    train_losses = []
    test_losses = []

    best_bac = 0

    for epoch in range(EPOCHS):
        print("Epoch", epoch)
        for j in range(1):
            for i in range(math.ceil(len(X_train)/BATCH_SIZE)):
                x_batch = X_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                y_batch = Y_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                mask_batch = mask_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                # print(x_batch.shape, y_batch.shape, mask_batch.shape)
            
                # if i == 0:
                #     print("X", x_batch.dtype)
                #     print("Y", y_batch.dtype)
                #     print("A", A_hat.dtype)
                #     print("mask", mask_batch.dtype)
                if epoch != 0:
                    batch_train_loss = f_train(x_batch, y_batch, A_hat, mask_batch)
        
        test_output = f_eval(X_test, A_hat)
        train_output = f_eval(X_train, A_hat)
    
        test_loss = cost_eval(X_test, Y_test, A_hat, mask_test)
        train_loss = cost_eval(X_train, Y_train, A_hat, mask_train)
    
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))
    
    
        test_predictions = test_output.argmax(-1).flatten()[mask_test.flatten().astype("bool")]
        real_classes = Y_test.argmax(-1).flatten()[mask_test.flatten().astype("bool")]
    
        conf = ConfusionMatrix(test_predictions, real_classes)

        if conf.bac() > best_bac:
            best_bac = conf.bac()
            best_pred = test_output.argmax(-1)
            best_output = test_output
            real_test_classes = real_classes
            best_epoch = epoch
    
        train_predictions = train_output.argmax(-1).flatten()[mask_train.flatten().astype("bool")]
        real_train_classes = Y_train.argmax(-1).flatten()[mask_train.flatten().astype("bool")]
    
        conf_train = ConfusionMatrix(train_predictions, real_train_classes)


    
    results = {
        "best_bac": best_bac,
        "best_epoch": best_epoch,
        "best_epoch_predictions": best_pred.tolist(),
        "best_epoch_output": best_output.tolist(),
        "real_classes": Y_test.tolist(),
        "data_order": order,
        "train_losses": train_losses,
        "test_losses": test_losses
    }

    results_obj.append(results)
    
run_number = "2"
run_number = str(run_number)  # Just to be sure

filename = "../Results_coli/" + run_number + "/" + "Train_GCN_3_results"
if shuffle:
    filename = filename + "_shuffle"
filename = filename + ".json"

with open(filename, "w") as outfile:
    json.dump(results_obj, outfile)
