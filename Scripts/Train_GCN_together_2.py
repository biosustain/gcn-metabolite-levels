# Train on all aa's simultaneously. Predict conc changes from all other concs and gene KO.
# All aa's have same GCN layer parameters, and individual dense layers

from __future__ import print_function

import argparse

import theano
import theano.tensor as T
import numpy as np
import lasagne
import math
import json
import random

helptext = ""

parser = argparse.ArgumentParser(description=helptext)

parser.add_argument('--share-fc', dest='sharefc', action='store_true')
parser.set_defaults(sharefc=False)

parser.add_argument("inputs", type=int, nargs="+", help="A list of integers corresponding to the input features that are included")
parser.add_argument("--no_coupling", action="store_true", help="Don't use the coupling network")
parser.add_argument("--shuffle", action="store_true", help="Shuffle the X data, creating random data. Useful for permutation testing.")
parser.add_argument("--suffix", type=str, help="Add this string to the name of the result files", default="")

args = parser.parse_args()

feature_list = args.inputs
no_coupling = args.no_coupling
shuffle = args.shuffle
suffix = args.suffix
sharefc = args.sharefc


class GCNLayer(lasagne.layers.Layer):
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
        
        self.W = self.add_param(W, [self.num_graphs, incoming.output_shape[2], num_out_features], name="W")
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_out_features)
    
    def get_output_for(self, input, **kwargs):
        A_X = T.tensordot(self.A_hat, input, axes=[1, 1]).transpose(2, 1, 0, 3)
        Z = T.tensordot(A_X, self.W, axes=2)
        return self.nonlinearity(Z)

  
def special_softmax(t):
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

#

# with open("../Data/aa_indices.json") as infile:
#     aa_indices = json.load(infile)
#     del aa_indices["cysteine"]

with open("../Data/Genome-wide/Measured_metabolite_indices.json") as infile:
    data_indices = json.load(infile)

loaded = np.load("../Data/NN_e_coli_data.npz")
data = dict(zip(loaded["arr_1"], loaded["arr_0"]))
del loaded

X = data["X"].astype("float32")
Y = data["Y"].astype("bool")

A_hat = data["adjacency"].astype("float32").transpose(2, 0, 1)
del data

splitpoint = int(X.shape[0] * 0.8)

order = random.sample(range(len(X)), len(X))

X = X[order]
Y = Y[order]

# Amplify knockout signal
X[:, :, 0] *= 3000

X_train = X[:splitpoint][:, :, feature_list]
X_test = X[splitpoint:][:, :, feature_list]

# Only predict measured metabolites
data_mask = np.zeros(shape=(Y.shape[0], Y.shape[1], 1))
data_mask[:, data_indices] = 1

Y = np.concatenate([(Y == 0).astype("int"), (Y == 1).astype("int")], axis=2)
Y = Y * data_mask

if no_coupling:
    A_hat = np.concatenate([A_hat[0], A_hat[2]])
print("Number of network structures:", len(A_hat))



# X_train = {}
Y_train = {}
# mask_train = {}

# X_test = {}
Y_test = {}
# mask_test = {}

#

all_train_cat_factors = {}

if shuffle:
    shuffle_seed = np.random.choice(range(1000))

for i, idx in enumerate(data_indices):
    if i % 50 == 0:
        print(i)
        print(sum([a.size * a.dtype.itemsize for a in Y_train.values()]) / 1e6)
    
    # X = data["X"].astype("float32").copy()
    # X = X[order]

    # Y = data["Y"].astype("float32").copy()
    # Y = np.concatenate([(Y == 0).astype("int"), (Y == 1).astype("int")], axis=2)

    # Y = Y[order]
    # cat_Y = loaded["cat_Y"].astype("float32")
    # covariates = Y.copy()
    
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(X)
    
    # Only predict one metabolite
    met_mask = np.zeros_like(Y)
    met_mask[:, idx] = 1
    met_Y = Y * met_mask

    # mask = (met_Y != 0).astype("float32")

    # covariates[:, idx] = 0
    # X = np.concatenate([X, covariates], axis=2)
    # X_train[idx] = X_train  # X[:splitpoint][:, :, feature_list]
    Y_train[idx] = met_Y[:splitpoint].astype("bool")[:, idx]
    # X_test[idx] = X_test  # X[splitpoint:][:, :, feature_list]
    Y_test[idx] = met_Y[splitpoint:].astype("bool")[:, idx]

    # mask_train[idx] = mask[:splitpoint]
    # mask_test[idx] = mask[splitpoint:]
    
    train_cat_counts = Y_train[idx].sum(0)
    train_cat_factors = [train_cat_counts.max() / cnt for cnt in train_cat_counts]
    all_train_cat_factors[idx] = train_cat_factors

    # for i, fact in enumerate(train_cat_factors):
    #     mask_train[idx][:, :, 0] *= (Y_train[idx][:, :, i] == 1).astype("int") * (fact - 1) + 1
        
first_met = data_indices[0]

NUM_INPUTS = X_train.shape[2]
NUM_OUTPUTS = Y_train[first_met].shape[-1]
print("NUM_INPUTS =", NUM_INPUTS)
print("NUM_OUTPUTS = ", NUM_OUTPUTS)
x_sym = T.tensor3("x_sym")
y_sym = T.matrix("y_sym")
A_sym = T.tensor3("A_sym")
ymask_sym = T.matrix("ymask_sym")

result_object = []

for num_round in range(1):

    l_in = lasagne.layers.InputLayer((None, A_hat.shape[0], NUM_INPUTS))
    l_1 = GCNLayer(l_in, A_sym, num_out_features=5, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 5
    l_2 = GCNLayer(l_1, A_sym, num_out_features=7, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 7
    l_3 = GCNLayer(l_2, A_sym, num_out_features=9, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 9
    l_4 = GCNLayer(l_3, A_sym, num_out_features=9, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 13
    l_5 = GCNLayer(l_4, A_sym, num_out_features=7, nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_concat = lasagne.layers.ConcatLayer([l_1, l_2, l_3, l_4], axis=2)

    met_output_layers = {}
    f_train = {}
    f_eval = {}
    cost_eval = {}

    if sharefc:
        l_slice = lasagne.layers.SliceLayer(l_concat, indices=data_indices[0], axis=1)
        l_dense = lasagne.layers.DenseLayer(l_slice, num_units=30, num_leading_axes=1, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        shared_l_dense_W = l_dense.W

        l_drop = lasagne.layers.DropoutLayer(l_dense)

        shared_l_out = lasagne.layers.DenseLayer(l_drop, num_units=NUM_OUTPUTS, num_leading_axes=1, nonlinearity=special_softmax)
        shared_l_out_W = shared_l_out.W

    for i, idx in enumerate(data_indices):
        # print(idx)
        if i % 10 == 0:
            print(i)
    
        if sharefc:
            # l_out = shared_l_out
            l_slice = lasagne.layers.SliceLayer(l_concat, indices=idx, axis=1)
            l_dense = lasagne.layers.DenseLayer(
                l_slice, num_units=30, num_leading_axes=1, nonlinearity=lasagne.nonlinearities.leaky_rectify,
                W=shared_l_dense_W
            )
            l_drop = lasagne.layers.DropoutLayer(l_dense)

            l_out = lasagne.layers.DenseLayer(
                l_drop, num_units=NUM_OUTPUTS, num_leading_axes=1, nonlinearity=special_softmax,
                W=shared_l_out_W
            )
        
        else:
            l_slice = lasagne.layers.SliceLayer(l_concat, indices=idx, axis=1)
            l_dense = lasagne.layers.DenseLayer(
                l_slice, num_units=20, num_leading_axes=1, nonlinearity=lasagne.nonlinearities.leaky_rectify
            )
            l_drop = lasagne.layers.DropoutLayer(l_dense)

            l_out = lasagne.layers.DenseLayer(
                l_drop, num_units=NUM_OUTPUTS, num_leading_axes=1, nonlinearity=special_softmax
            )

            # print("Dense layer",
            #       lasagne.layers.get_output(l_dense, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)
            # print("Output layer",
            #       lasagne.layers.get_output(l_out, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat}).shape)

        met_output_layers[idx] = l_out
    
        # Retrieve network output
        train_out = lasagne.layers.get_output(l_out, inputs={l_in: x_sym}, deterministic=False)
        eval_out = lasagne.layers.get_output(l_out, inputs={l_in: x_sym}, deterministic=True)

        all_params = lasagne.layers.get_all_params(l_out, trainable=True)

        # cost = T.nnet.categorical_crossentropy(train_out+1e-8, y_sym).mean()
        # cost = lasagne.objectives.squared_error(train_out, y_sym)

        cost = lasagne.objectives.categorical_crossentropy(train_out+1e-8, y_sym)  # [:, idx])
        cost = lasagne.objectives.aggregate(cost, weights=ymask_sym[:, 0], mode="mean")

        # cost = lasagne.objectives.categorical_crossentropy(train_out+1e-8, y_sym)
        # cost = lasagne.objectives.aggregate(cost, weights=ymask_sym[:, :, 0], mode="mean")

        #cost_eval = lasagne.objectives.squared_error(eval_out, y_sym)

        all_grads = T.grad(cost, all_params)

        updates = lasagne.updates.adamax(all_grads, all_params, learning_rate=0.06)  # learning_rate=0.002) #, learning_rate=0.002)

        f_eval[idx] = theano.function([x_sym, A_sym],
                             eval_out, on_unused_input='warn')

        cost_eval[idx] = theano.function([x_sym, y_sym, A_sym, ymask_sym],
                             cost, on_unused_input='warn')

        f_train[idx] = theano.function(
            [x_sym, y_sym, A_sym, ymask_sym],
            cost,
            updates=updates, on_unused_input='warn'
        )

    train_losses = []
    test_losses = []

    bacs_train = {}
    bacs_test = {}

    best_bac = 0

    EPOCHS = 30
    BATCH_SIZE = 100

    for epoch in range(EPOCHS):
        print("Starting epoch", epoch)
        epoch_train_losses = []
        epoch_test_losses = []
        epoch_test_bacs = {}
        epoch_predictions = {}
        for j in range(1):
            # print("------", math.ceil(len(X_train[idx])/BATCH_SIZE), len(X_train[idx]), BATCH_SIZE)
            for i in range(math.ceil(len(X_train)/BATCH_SIZE)):
                # print("  Training batch", i)
                for idx in random.sample(data_indices, len(data_indices)):
                    # print("    Training for met", idx)

                    x_batch = X_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    y_batch = Y_train[idx].astype("float32")[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    # mask_batch = mask_train[idx][i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    mask_batch = (y_batch != 0).astype("float32")

                    for i, fact in enumerate(train_cat_factors):
                        mask_batch[:, 0] *= (y_batch[:, i] == 1).astype("int") * (fact - 1) + 1

                    if epoch != 0:
                        batch_train_loss = f_train[idx](x_batch, y_batch, A_hat, mask_batch)

        print("  Done training in epoch", epoch)
        for idx in data_indices:
            print(idx)
            mask_train = np.ones_like(Y_train[idx].astype("float32"))
            #mask_train[:, idx] = 1

            mask_test = np.ones_like(Y_test[idx].astype("float32"))
            #mask_test[:, idx] = 1

            test_output = f_eval[idx](X_test, A_hat)
            train_output = f_eval[idx](X_train, A_hat)

            test_loss = cost_eval[idx](X_test, Y_test[idx].astype("float32"), A_hat, mask_test)
            train_loss = cost_eval[idx](X_train, Y_train[idx].astype("float32"), A_hat, mask_train)
    
            epoch_train_losses.append(train_loss)
            epoch_test_losses.append(test_loss)
    
            test_predictions = test_output.argmax(-1)
            # print("Test prediction shape:", test_predictions.shape)
            real_test_classes = Y_test[idx].astype("float32").argmax(-1)
    
            conf = ConfusionMatrix(test_predictions, real_test_classes)
            bacs_test[idx] = conf.bac()
            epoch_test_bacs[idx] = conf.bac()
            epoch_predictions[idx] = list(test_predictions.astype("object"))
    
            train_predictions = train_output.argmax(-1)
            real_classes = Y_train[idx].astype("float32").argmax(-1)

            conf = ConfusionMatrix(train_predictions, real_classes)
            bacs_train[idx] = conf.bac()
        
        train_losses.append(sum(epoch_train_losses)/len(epoch_train_losses))
        test_losses.append(sum(epoch_test_losses)/len(epoch_test_losses))
    
        epoch_bac = sum(epoch_test_bacs.values())/len(epoch_test_bacs)
        if epoch_bac > best_bac:
            best_bac = epoch_bac
            best_epoch_bacs = epoch_test_bacs
            best_epoch_predictions = epoch_predictions
            best_epoch = epoch
    
    result_object.append({
        "best_bac": best_bac,
        "best_epoch": best_epoch,
        "best_epoch_bacs": best_epoch_bacs,
        "best_epoch_predictions": best_epoch_predictions
    })


run_number = "1"
feature_string = "-".join(map(str, feature_list))

if no_coupling:
    feature_string += "_nocoupling"
    
if shuffle:
    feature_string += "_shuffle"
    
if suffix:
    feature_string += "_" + suffix
    
if sharefc:
    feature_string += "_shared"
    
with open("../Results_coli/" + run_number + "/" + feature_string + "_results.json", "w") as outfile:
    json.dump(result_object, outfile)

    






