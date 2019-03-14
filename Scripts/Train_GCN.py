from __future__ import print_function

import argparse

import theano
import theano.tensor as T
import numpy as np
import lasagne
import math
import json

helptext = """
Script for training neural networks on the HPC cluster

Categorical concentration change predictions (yes/no) for a single amino acid will be trained on one or several of:
 - 0: Knockouts
 - 1: Node type (metabolite/reaction)
 - 2: WT fluxes
 - 3: Predicted KO fluxes
 - 4: Concentrations of the remaining amino acids

"""

parser = argparse.ArgumentParser(description=helptext)

parser.add_argument("inputs", type=int, nargs="+", help="A list of integers corresponding to the input features that are included")
parser.add_argument("--no_coupling", action="store_true", help="Don't use the coupling network")
parser.add_argument("--shuffle", action="store_true", help="Shuffle the X data, creating random data. Useful for permutation testing.")
parser.add_argument("--suffix", type=str, help="Add this string to the name of the result files", default="")

args = parser.parse_args()

feature_list = args.inputs
no_coupling = args.no_coupling
shuffle = args.shuffle
suffix = args.suffix


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
    print("Not using coupling")
    A_hat_2 = A_hat_2[:1]
    
num_graphs = len(A_hat_2)
print("Number of network structures:", num_graphs)

data_fields = ("X_train", "Y_train", "X_test", "Y_test", "mask_train", "mask_test")
aa_datasets = {}
for aa in aa_indices:
    aa_data = np.load("../Data/AA_datasets/" + aa + ".npz")
    aa_datasets[aa] = {
        key: aa_data[key].astype("float32") for key in data_fields
    }



bacs_and_epochs = {}
outputs_and_predictions = {}
for aa in aa_indices:
    print(aa)
    results = []
    results_output = []

    for training_round in range(5):
        print("  ", training_round)
        NUM_INPUTS = len(feature_list)
        NUM_OUTPUTS = 2

        x_sym = T.tensor3("x_sym")
        y_sym = T.tensor3("y_sym")
        A_sym = T.tensor3("A_sym")
        ymask_sym = T.tensor3("ymask_sym")

        l_in = lasagne.layers.InputLayer((None, A_hat_2.shape[0], NUM_INPUTS))

        l_1 = GCNLayer(l_in, A_sym, num_out_features=3, num_graphs=num_graphs, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 5
        l_2 = GCNLayer(l_1, A_sym, num_out_features=5, num_graphs=num_graphs, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 7
        l_3 = GCNLayer(l_2, A_sym, num_out_features=7, num_graphs=num_graphs, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 9
        l_4 = GCNLayer(l_3, A_sym, num_out_features=9, num_graphs=num_graphs, nonlinearity=lasagne.nonlinearities.leaky_rectify) # 13
        l_5 = GCNLayer(l_4, A_sym, num_out_features=7, num_graphs=num_graphs, nonlinearity=lasagne.nonlinearities.leaky_rectify)

        l_concat = lasagne.layers.ConcatLayer([l_1, l_2, l_3, l_4], axis=2)

        # Dense: 50 units, leaky_rectify
        l_slice = lasagne.layers.SliceLayer(l_concat, indices=aa_indices[aa], axis=1)
        l_dense = lasagne.layers.DenseLayer(l_slice, num_units=30, num_leading_axes=1, nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_drop = lasagne.layers.DropoutLayer(l_dense)

        l_out = lasagne.layers.DenseLayer(l_drop, num_units=NUM_OUTPUTS, num_leading_axes=1, nonlinearity=special_softmax)


        all_params = lasagne.layers.get_all_params(l_out, trainable=True)

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

        all_grads = T.grad(cost, all_params)

        updates = lasagne.updates.adamax(all_grads, all_params, learning_rate=0.06)#, learning_rate=0.002) #, learning_rate=0.002)

        f_eval = theano.function([x_sym, A_sym],
                             eval_out, on_unused_input='warn')

        cost_eval = theano.function([x_sym, y_sym, A_sym, ymask_sym],
                             cost, on_unused_input='warn')

        f_train = theano.function(
            [x_sym, y_sym, A_sym, ymask_sym],
            cost,
            updates=updates, on_unused_input='warn'
        )

        EPOCHS = 50
        BATCH_SIZE = 10

        train_losses = []
        test_losses = []

        best_bac = 0
        
        dataset = aa_datasets[aa]
        X_train = dataset["X_train"][:, :, feature_list]
        Y_train = dataset["Y_train"]
        X_test = dataset["X_test"][:, :, feature_list]
        Y_test = dataset["Y_test"]
        mask_train = dataset["mask_train"]
        mask_test = dataset["mask_test"]
        
        if shuffle:
            print("SHUFFLING!")
            np.random.shuffle(X_train)
            np.random.shuffle(X_test)

        for epoch in range(EPOCHS):
            for j in range(1):
                for i in range(math.ceil(len(X_train)/BATCH_SIZE)):
                    x_batch = X_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    y_batch = Y_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    mask_batch = mask_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]

                    if epoch != 0:
                        batch_train_loss = f_train(x_batch, y_batch, A_hat_2, mask_batch)

            test_output = f_eval(X_test, A_hat_2)
            train_output = f_eval(X_train, A_hat_2)

            test_loss = cost_eval(X_test, Y_test, A_hat_2, mask_test)
            train_loss = cost_eval(X_train, Y_train, A_hat_2, mask_train)

            train_losses.append(train_loss)
            test_losses.append(test_loss)


            test_predictions = test_output.argmax(-1)#.flatten()[mask_test.flatten().astype("bool")]
            real_classes = Y_test.argmax(-1)[:, aa_indices[aa]]#.flatten()[mask_test.flatten().astype("bool")]

            conf = ConfusionMatrix(test_predictions, real_classes)

            if conf.bac() > best_bac:
                best_bac = conf.bac()
                best_pred = test_predictions
                best_output = test_output
                real_test_classes = real_classes
                best_epoch = epoch

            test_predictions = train_output.argmax(-1)#.flatten()[mask_train.flatten().astype("bool")]
            real_classes = Y_train.argmax(-1)[:, aa_indices[aa]]#.flatten()[mask_train.flatten().astype("bool")]

            conf = ConfusionMatrix(test_predictions, real_classes)


        results.append({
            "best_bac": best_bac,
            "best_epoch": best_epoch
        })
        results_output.append({
            "best_pred": best_pred,
            "best_output": best_output,
            "real_test_classes": real_test_classes,
        })
    bacs_and_epochs[aa] = results
    outputs_and_predictions[aa] = results_output

# Save results to files
run_number = "1"

feature_string = "-".join(map(str, feature_list))

if no_coupling:
    feature_string += "_nocoupling"
    
if shuffle:
    feature_string += "_shuffle"
    
if suffix:
    feature_string += "_" + suffix

with open("../Results/" + run_number + "/" + feature_string + "_results.json", "w") as outfile:
    json.dump(bacs_and_epochs, outfile)
    
numpy_dict = {}
for aa in outputs_and_predictions:
    for i, di in enumerate(outputs_and_predictions[aa]):
        for key, value in di.items():
            key_string = "-".join((aa, str(i), key))
            numpy_dict[key_string] = value
np.save("../Results/" + run_number + "/" + feature_string + "_outputs.npy", numpy_dict)
    

    