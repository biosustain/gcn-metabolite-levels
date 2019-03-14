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
        self, incoming, A_hat, num_out_features=3, num_graphs=2,
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

with open("../Data/aa_indices.json") as infile:
    aa_indices = json.load(infile)
    del aa_indices["cysteine"]
        
data = np.load("../Data/NN_data.npz")
data = dict(zip(data["arr_1"], data["arr_0"]))
A_hat_2 = data["double_A"].astype("float32")

if no_coupling:
    A_hat_2 = A_hat_2[:1]
print("Number of network structures:", len(A_hat_2))

loaded = np.load("../Data/Permuted_data1.npz")

splitpoint = 75

X_train = {}
Y_train = {}
mask_train = {}

X_test = {}
Y_test = {}
mask_test = {}

if shuffle:
    shuffle_seed = np.random.choice(range(1000))

for aa, idx in aa_indices.items():
    
    X = loaded["X"].astype("float32").copy()
    Y = loaded["Y"].astype("float32")
    cat_Y = loaded["cat_Y"].astype("float32")
    covariates = Y.copy()
    
    if shuffle:
        np.random.seed(shuffle_seed)
        np.random.shuffle(X)
    
    # Only predict one metabolite
    aa_mask = np.zeros_like(Y)
    aa_mask[:, aa_indices[aa]] = 1
    Y = Y * aa_mask
    
    mask = (Y != 0).astype("float32")
    cat_Y = cat_Y * mask

    covariates[:, aa_indices[aa]] = 0
    X = np.concatenate([X, covariates], axis=2)
    
    # Amplify knockout signal
    X[:, :, 0] *= 3000
    
    Y = cat_Y
    
    X_train[aa] = X[:splitpoint][:, :, feature_list]
    Y_train[aa] = Y[:splitpoint]
    X_test[aa] = X[splitpoint:][:, :, feature_list]
    Y_test[aa] = Y[splitpoint:]

    mask_train[aa] = mask[:splitpoint]
    mask_test[aa] = mask[splitpoint:]
    
    train_cat_counts = Y_train[aa].sum(1).sum(0)
    train_cat_factors = [train_cat_counts.max() / cnt for cnt in train_cat_counts]

    for i, fact in enumerate(train_cat_factors):
        mask_train[aa][:, :, 0] *= (Y_train[aa][:, :, i] == 1).astype("int") * (fact - 1) + 1
        
first_aa = list(aa_indices)[0]
        
NUM_INPUTS = X_train[first_aa].shape[2]
NUM_OUTPUTS = Y_train[first_aa].shape[-1]
x_sym = T.tensor3("x_sym")
y_sym = T.tensor3("y_sym")
A_sym = T.tensor3("A_sym")
ymask_sym = T.tensor3("ymask_sym")

result_object = []

for num_round in range(5):

    l_in = lasagne.layers.InputLayer((None, A_hat_2.shape[0], NUM_INPUTS))
    l_1 = GCNLayer(l_in, A_sym, num_out_features=5, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 5
    l_2 = GCNLayer(l_1, A_sym, num_out_features=7, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 7
    l_3 = GCNLayer(l_2, A_sym, num_out_features=9, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 9
    l_4 = GCNLayer(l_3, A_sym, num_out_features=9, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 13
    l_5 = GCNLayer(l_4, A_sym, num_out_features=7, nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_concat = lasagne.layers.ConcatLayer([l_1, l_2, l_3, l_4, l_5], axis=2)

    aa_output_layers = {}
    f_train = {}
    f_eval = {}
    cost_eval = {}



    if sharefc:
        l_slice = lasagne.layers.SliceLayer(l_concat, indices=idx, axis=1)
        l_dense = lasagne.layers.DenseLayer(l_slice, num_units=30, num_leading_axes=1, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_drop = lasagne.layers.DropoutLayer(l_dense)

        shared_l_out = lasagne.layers.DenseLayer(l_drop, num_units=NUM_OUTPUTS, num_leading_axes=1, nonlinearity=special_softmax)

    for aa, idx in list(aa_indices.items())[:1]:
        print(aa)
    
        if sharefc:
            l_out = shared_l_out
        
        else:
            l_slice = lasagne.layers.SliceLayer(l_concat, indices=idx, axis=1)
            l_dense = lasagne.layers.DenseLayer(l_slice, num_units=30, num_leading_axes=1, nonlinearity=lasagne.nonlinearities.leaky_rectify)
            l_drop = lasagne.layers.DropoutLayer(l_dense)

            l_out = lasagne.layers.DenseLayer(l_drop, num_units=NUM_OUTPUTS, num_leading_axes=1, nonlinearity=special_softmax)
            
            print(X[:100].shape)
            print("Dense layer",
                lasagne.layers.get_output(l_dense, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat_2}).shape)
            print("Output layer",
                lasagne.layers.get_output(l_out, inputs={l_in: x_sym}).eval({x_sym: X[:100], A_sym: A_hat_2}).shape)
    
        aa_output_layers[aa] = l_out
    
        # Retrieve network output
        train_out = lasagne.layers.get_output(l_out, inputs={l_in: x_sym}, deterministic=False)
        eval_out = lasagne.layers.get_output(l_out, inputs={l_in: x_sym}, deterministic=True)

        all_params = lasagne.layers.get_all_params(l_out, trainable=True)

        # cost = T.nnet.categorical_crossentropy(train_out+1e-8, y_sym).mean()
        # cost = lasagne.objectives.squared_error(train_out, y_sym)
        cost = lasagne.objectives.categorical_crossentropy(train_out+1e-8, y_sym[:, aa_indices[aa]])
        cost = lasagne.objectives.aggregate(cost, weights=ymask_sym[:, aa_indices[aa], 0], mode="mean")

        # cost = lasagne.objectives.categorical_crossentropy(train_out+1e-8, y_sym)
        # cost = lasagne.objectives.aggregate(cost, weights=ymask_sym[:, :, 0], mode="mean")

        #cost_eval = lasagne.objectives.squared_error(eval_out, y_sym)

        all_grads = T.grad(cost, all_params)

        updates = lasagne.updates.adamax(all_grads, all_params, learning_rate=0.06)#, learning_rate=0.002) #, learning_rate=0.002)

        f_eval[aa] = theano.function([x_sym, A_sym],
                             eval_out, on_unused_input='warn')

        cost_eval[aa] = theano.function([x_sym, y_sym, A_sym, ymask_sym],
                             cost, on_unused_input='warn')

        f_train[aa] = theano.function(
            [x_sym, y_sym, A_sym, ymask_sym],
            cost,
            updates=updates, on_unused_input='warn'
        )
	
    train_losses = []
    test_losses = []

    bacs_train = {}
    bacs_test = {}

    best_bac = 0

    EPOCHS = 50
    BATCH_SIZE = 10

    for epoch in range(EPOCHS):
        epoch_train_losses = []
        epoch_test_losses = []
        epoch_test_bacs = {}
        epoch_predictions = {}
        for j in range(1):
            for i in range(math.ceil(len(X_train[aa])/BATCH_SIZE)):
                for aa in random.sample(list(aa_indices), len(aa_indices)):
                    x_batch = X_train[aa][i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    y_batch = Y_train[aa][i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    mask_batch = mask_train[aa][i*BATCH_SIZE: (i+1)*BATCH_SIZE]

                    if epoch != 0:
                        batch_train_loss = f_train[aa](x_batch, y_batch, A_hat_2, mask_batch)

        for aa in sorted(aa_indices):
            test_output = f_eval[aa](X_test[aa], A_hat_2)
            train_output = f_eval[aa](X_train[aa], A_hat_2)

            test_loss = cost_eval[aa](X_test[aa], Y_test[aa], A_hat_2, mask_test[aa])
            train_loss = cost_eval[aa](X_train[aa], Y_train[aa], A_hat_2, mask_train[aa])
    
            epoch_train_losses.append(train_loss)
            epoch_test_losses.append(test_loss)
    
            test_predictions = test_output.argmax(-1)
            real_test_classes = Y_test[aa].argmax(-1)[:, aa_indices[aa]]
    
            conf = ConfusionMatrix(test_predictions, real_test_classes)
            bacs_test[aa] = conf.bac()
            epoch_test_bacs[aa] = conf.bac()
            epoch_predictions[aa] = test_predictions
    
            train_predictions = train_output.argmax(-1)
            real_classes = Y_train[aa].argmax(-1)[:, aa_indices[aa]]

            conf = ConfusionMatrix(train_predictions, real_classes)
            bacs_train[aa] = conf.bac()
        
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
        "best_epoch_bacs": best_epoch_bacs
    })


run_number = "3"
feature_string = "-".join(map(str, feature_list))

if no_coupling:
    feature_string += "_nocoupling"
    
if shuffle:
    feature_string += "_shuffle"
    
if suffix:
    feature_string += "_" + suffix
    
if sharefc:
    feature_string += "_shared"
    
with open("../Results/" + run_number + "/" + feature_string + "_results.json", "w") as outfile:
    json.dump(result_object, outfile)

    






