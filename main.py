# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:17:12 2015

@author: sakurai
"""

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import chainer.functions as F
from chainer import optimizers
from chainer import Variable, FunctionSet
import spatial_transformer_network as stm


def forward(model, x_batch, train=False):
    x = Variable(x_batch, volatile=not train)
    x_st, theta = model.st(x)
    h = F.relu(model.fc1(x_st))
    h = F.dropout(h, train=train)
    h = F.relu(model.fc2(h))
    h = F.dropout(h, train=train)
    y = model.fc3(h)
    return y, x_st, theta

if __name__ == '__main__':
    try:
        x_train_data
    except NameError:
        (x_train_data, t_train_data,
         x_valid_data, t_valid_data,
         x_test_data, t_test_data) = stm.load_cluttered_mnist()

    num_train = len(x_train_data)
    num_valid = len(x_valid_data)
    num_test = len(x_test_data)
    in_shape = x_train_data.shape[1:]
    out_shape = (28, 28)
    out_size = np.prod(out_shape)  # 784

#    x_train = Variable(x_train_data)
#    t_train = Variable(t_train_data)
#    x_valid = Variable(x_valid_data)
#    t_valid = Variable(t_valid_data)
#    x_test = Variable(x_test_data)
#    t_test = Variable(t_test_data)

    model = FunctionSet(st=stm.SpatialTransformer(in_shape, out_shape,
                                                  "affine"),
                        fc1=F.Linear(out_size, 256),
                        fc2=F.Linear(256, 256),
                        fc3=F.Linear(256, 10))
    initial_model = copy.deepcopy(model)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    batch_size = 250
    num_batches = num_train / batch_size
    max_epochs = 1000
    l2_reg = 0.000001

    train_loss_history = []
    train_accuracy_history = []
    valid_loss_history = []
    valid_accuracy_history = []
    valid_loss_best = 100
    valid_accuracy_best = 0
    epoch_best = 0
    try:
        for epoch in xrange(max_epochs):
            print "epoch", epoch,
            time_begin = time.clock()
            losses = []
            accuracies = []
            gWs = 0
            perm = np.random.permutation(num_train)
            for indices in np.array_split(perm, num_batches):
                x_batch = x_train_data[indices]
                t_batch = t_train_data[indices]
                optimizer.zero_grads()
                y, x_st, theta = forward(model, x_batch, train=True)
                t = Variable(t_batch)
                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)
                losses.append(loss.data)
                accuracies.append(accuracy.data)
                loss.backward()
#                optimizer.weight_decay(l2_reg)
#                optimizer.clip_grads(500)
                optimizer.update()
                gWs += np.array([np.linalg.norm(w) for w in
                                 model.gradients[::2]])

            train_loss = np.mean(losses)
            train_accuracy = np.mean(accuracies)
            y_valid, x_st_valid, theta_valid = forward(model, x_valid_data)
            t_valid = Variable(t_valid_data, volatile=True)
            valid_loss = F.softmax_cross_entropy(y_valid, t_valid).data
            valid_accuracy = F.accuracy(y_valid, t_valid).data

            if valid_loss < valid_loss_best:
                model_best = copy.deepcopy(model)
                valid_loss_best = valid_loss
                valid_accuracy_best = valid_accuracy
                epoch_best = epoch
                print "(Best score!)",
            print "(time: %f)" % (time.clock() - time_begin)

            # print norms of the weights
            print "    |W|", [np.linalg.norm(w) for w in
                              model.parameters[::2]]
            print "    |gW|", gWs.astype(np.float32).tolist()

            # pring scores
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            valid_loss_history.append(valid_loss)
            valid_accuracy_history.append(valid_accuracy)
            print "    [train] loss: %f" % train_loss
            print "    [valid] loss: %f" % valid_loss
            print "    [valid] best loss: %f (at #%d)" % (valid_loss_best,
                                                          epoch_best)
            print "    [train] accuracy: %f" % train_accuracy
            print "    [valid] accuracy: %f" % valid_accuracy
            print "    [valid] best accuracy: %f (at #%d)" % (
                valid_accuracy_best, epoch_best)

            # plot loss histories
            fig = plt.figure()
            plt.plot(np.arange(epoch+1), np.array(train_loss_history))
            plt.plot(np.arange(epoch+1), np.array(valid_loss_history), '-g')
            plt.plot([0, epoch+1], [valid_loss_best]*2, '-g')
            plt.ylabel('loss')
            plt.ylim([0, 2])
            plt.legend(['tloss', 'vloss'],
                       loc='best')
            # plot accuracy histories
            plt.twinx()
            plt.plot(np.arange(epoch+1), np.array(train_accuracy_history))
            plt.plot(np.arange(epoch+1), np.array(valid_accuracy_history),
                     'r-')
            plt.plot([0, epoch+1], [valid_accuracy_best]*2, 'r-')
            plt.ylabel('accuracy')
            plt.ylim([0.6, 1])

            plt.legend(['tacc', 'vacc'],
                       loc='best')
            plt.plot([epoch_best]*2, [0, 1], '-k')
            plt.grid()
            plt.show()
            plt.draw()

            print "model.theta.bias:", model.st.parameters[-1]
#            print "theta:", theta.data[0]
            print "theta:", theta_valid.data[0]
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
#            ax.matshow(x_batch[0].reshape(in_shape), cmap=plt.cm.gray)
            ax.matshow(x_valid_data[0].reshape(in_shape), cmap=plt.cm.gray)
            ax = fig.add_subplot(1, 2, 2)
#            ax.matshow(x_st.data[0].reshape(out_shape), cmap=plt.cm.gray)
            ax.matshow(x_st_valid.data[0].reshape(out_shape), cmap=plt.cm.gray)
            plt.show()
            plt.draw()

    except KeyboardInterrupt:
        pass
