import theano
from pylearn2.models import mlp
from pylearn2.train_extensions import best_params
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.utils import serial
from pylearn2.termination_criteria import MonitorBased
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import randint
import itertools
import os

scaler = StandardScaler()

class KGL(DenseDesignMatrix):

    def __init__(self, filename, X=None, Y=None, count=0):

        if X == None:
            self.class_names = ['0', '1']
            X = []
            Y = []
            with open(filename) as fin:
                first_line = True
                i = 0
                counter = 0
                for line in fin:
                    i += 1
                    if i == 1:
                        continue
                    if first_line:
                        X.append([float(n) for n in line.split()])
                    else:
                        Y.append([int(n) for n in line.split()])
                    first_line = not first_line
                    if count > 0 and counter == count - 1:
						break
                    counter += 1

            X = np.array(X)
            Y = np.array(Y)

        super(KGL, self).__init__(X=X, y=Y)

    @property
    def nr_inputs(self):
        return len(self.X[0])

    def split(self, prop=.8):
        cutoff = int(len(self.y) * prop)
        X1, X2 = self.X[:cutoff], self.X[cutoff:]
        y1, y2 = self.y[:cutoff], self.y[cutoff:]
        return KGL(None, X1, y1), KGL(None, X2, y2)

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        return itertools.izip_longest(self.X, self.y)


# create dataset
ds_train = KGL('../../../data/fann_sentiments.10k.train', count=100)
ds_train, ds_valid = ds_train.split(0.7)
ds_valid, ds_test = ds_valid.split(0.7)
# create hidden layer with 2 nodes, init weights in range -0.1 to 0.1 and add
# a bias with value 1
hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=500, irange=.1, init_bias=1.)
# create Softmax output layer
output_layer = mlp.Softmax(5, 'output', irange=.1)

layers = [hidden_layer, output_layer]

# termination_criterion = EpochCounter(400)
# termination criterion that stops after 50 epochs without
# any increase in misclassification on the validation set
termination_criterion = MonitorBased(channel_name='output_misclass',
                                     N=500, prop_decrease=0.0)

# momentum
initial_momentum = .5
final_momentum = .99
start = 1
saturate = 50
momentum_adjustor = learning_rule.MomentumAdjustor(final_momentum, start, saturate)
momentum_rule = learning_rule.Momentum(initial_momentum)

# learning rate
start = 1
saturate = 50
decay_factor = .1
learning_rate_adjustor = sgd.LinearDecayOverEpoch(start, saturate, decay_factor)

# create neural net
ann = mlp.MLP(layers, nvis=ds_train.nr_inputs)

# create Stochastic Gradient Descent trainer
trainer = sgd.SGD(learning_rate=.05, batch_size=10, monitoring_dataset=ds_valid,
                  termination_criterion=termination_criterion, learning_rule=momentum_rule)
trainer.setup(ann, ds_train)

# add monitor for saving the model with best score
monitor_save_best = best_params.MonitorBasedSaveBest('output_misclass',
                                                     '../../../data/best.pkl')

# train neural net until the termination criterion is true
while True:
    trainer.train(dataset=ds_train)
    ann.monitor.report_epoch()
    ann.monitor()
    monitor_save_best.on_monitor(ann, ds_valid, trainer)
    if not trainer.continue_learning(ann):
        break
    momentum_adjustor.on_monitor(ann, ds_valid, trainer)
    learning_rate_adjustor.on_monitor(ann, ds_valid, trainer)

# load the best model
ann = serial.load('/home/andrej/prj/kaggl_sent/data/best.pkl')

# function for classifying a input vector
def classify(inp):
    inp = np.asarray(inp)
    inp.shape = (1, ds_train.nr_inputs)
    return np.argmax(ann.fprop(theano.shared(inp, name='inputs')).eval())

# function for calculating and printing the models accuracy on a given dataset
def score(dataset):
    nr_correct = 0
    for features, label in dataset:
        if classify(features) == np.argmax(label):
            nr_correct += 1

    print '%s/%s correct' % (nr_correct, len(dataset))

print
print 'Accuracy of train set:'
score(ds_train)
print 'Accuracy of validation set:'
score(ds_valid)
print 'Accuracy of test set:'
score(ds_test)