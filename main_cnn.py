# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:17:12 2015

@author: sakurai
"""

import argparse
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import chainer.functions as F
from chainer import optimizers
from chainer import Variable, Chain, ChainList
from chainer import cuda
import spatial_transformer_network as stm

np.random.seed(0)


class CNNLocalizationNetwork(ChainList):
    def __init__(self, in_size, theta_size):
        sqrt2 = np.sqrt(2)
        super(CNNLocalizationNetwork, self).__init__(
            F.Convolution2D(1, 20, 5, wscale=sqrt2),
            F.Convolution2D(20, 20, 5, wscale=sqrt2),
            F.Linear(1620, 20, wscale=sqrt2),
            F.Linear(20, theta_size,
                     initialW=np.zeros((theta_size, 20), dtype=np.float32))
        )

    def __call__(self, x, train=False):
        if len(x.data.shape) == 3:
            shape = x.data.shape
            new_shape = shape[0:1] + (1,) + shape[1:]
            x = F.reshape(x, new_shape)
        h = F.max_pooling_2d(x, 2)
        h = F.max_pooling_2d(self[0](h), 2)
        h = F.relu(h)
        h = F.relu(self[1](h))
        h = F.relu(self[2](h))
        theta = self[3](h)  # theta has the shape of (len(x), theta_size)
        return theta


class SpatialTransformerNetworkCNN(Chain):
    def __init__(self, in_shape, out_shape, trans_type="translation"):
        assert trans_type in ["translation", "affine"]
        sqrt2 = np.sqrt(2)
        super(SpatialTransformerNetworkCNN, self).__init__(
            st=stm.SpatialTransformer(in_shape, out_shape, "affine",
                                      loc_net_class=CNNLocalizationNetwork),
            conv1=F.Convolution2D(1, 64, 9, wscale=sqrt2),
            conv2=F.Convolution2D(64, 64, 7, wscale=sqrt2),
            fc3=F.Linear(256, 10, wscale=sqrt2)
        )

    def forward(self, x_batch, train=False):
        x = Variable(x_batch, volatile=not train)
        x_st, theta, points = self.st(x, True)
        if len(x_st.data.shape) == 2:
            shape = x_st.data.shape
            new_shape = shape[0:1] + (1,) + (28, 28)
            x_st = F.reshape(x_st, new_shape)
        h = F.max_pooling_2d(self.conv1(x_st), 2)
        h = F.relu(h)
        h = F.dropout(h, train=train)
        h = F.max_pooling_2d(self.conv2(h), 2)
        h = F.relu(h)
        h = F.dropout(h, train=train)
        y = self.fc3(h)
        return y, x_st, theta, points

    def compute_loss(self, x_batch, t_batch, train=False,
                     return_variables=False):
        y, x_st, theta, points = self.forward(x_batch, train=train)
        t = Variable(t_batch, volatile=not train)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        if return_variables:
            return (loss, accuracy, (y, x_st, theta, points))
        else:
            return (loss, accuracy)


if __name__ == '__main__':
    try:
        x_train_data
    except NameError:
        (x_train_data, t_train_data,
         x_valid_data, t_valid_data,
         x_test_data, t_test_data) = stm.load_cluttered_mnist()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    x_valid_data = cuda.to_cpu(x_valid_data)
    t_valid_data = cuda.to_cpu(t_valid_data)
    x_test_data = cuda.to_cpu(x_test_data)
    t_test_data = cuda.to_cpu(t_test_data)
    num_train = len(x_train_data)
    num_valid = len(x_valid_data)
    num_test = len(x_test_data)
    in_shape = x_train_data.shape[1:]
    out_shape = (28, 28)
    out_size = np.prod(out_shape)  # 784

    model = SpatialTransformerNetworkCNN(in_shape, out_shape, "affine")
    if args.gpu >= 0:
        model.to_gpu()
        x_test_data = cuda.to_gpu(x_test_data)
        t_test_data = cuda.to_gpu(t_test_data)
    initial_model = copy.deepcopy(model)
    optimizer = optimizers.Adam(0.0005)
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
            time_begin = time.time()
            losses = []
            accuracies = []
            gWs = 0
            perm = np.random.permutation(num_train)
            for indices in np.array_split(perm, num_batches):
                x_batch = xp.asarray(x_train_data[indices])
                t_batch = xp.asarray(t_train_data[indices])
                loss, accuracy, variables = model.compute_loss(
                    x_batch, t_batch, train=True, return_variables=True)
                y, x_st, theta, points = variables
                optimizer.zero_grads()
                loss.backward()
#                optimizer.weight_decay(l2_reg)
#                optimizer.clip_grads(500)
                optimizer.update()

                losses.append(cuda.to_cpu(loss.data))
                accuracies.append(cuda.to_cpu(accuracy.data))
                gWs += xp.array([np.linalg.norm(cuda.to_cpu(p.data)) for p in
                                 model.params()])
            train_loss = np.mean(losses)
            train_accuracy = np.mean(accuracies)

            losses = []
            accuracies = []
            for indices in np.array_split(np.arange(num_valid), num_batches):
                x_batch = xp.asarray(x_valid_data[indices])
                t_batch = xp.asarray(t_valid_data[indices])
                actual_size = len(indices)
                loss, accuracy, variables = model.compute_loss(
                    x_batch, t_batch, train=False, return_variables=True)
                y, x_st, theta, points = variables

                losses.append(cuda.to_cpu(loss.data) * actual_size)
                accuracies.append(cuda.to_cpu(accuracy.data) * actual_size)
            valid_loss = np.sum(losses) / num_valid
            valid_accuracy = np.sum(accuracies) / num_valid

            if valid_loss < valid_loss_best:
                model_best = copy.deepcopy(model)
                valid_loss_best = valid_loss
                valid_accuracy_best = valid_accuracy
                epoch_best = epoch
                print "(Best score!)",
            print "(time: %f)" % (time.time() - time_begin)

            # print norms of the weights
            print "    |W|", [np.linalg.norm(cuda.to_cpu(p.data)) for p in
                              model.params()]
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
                       loc='lower left')
            # plot accuracy histories
            plt.twinx()
            plt.plot(np.arange(epoch+1), np.array(train_accuracy_history))
            plt.plot(np.arange(epoch+1), np.array(valid_accuracy_history),
                     'r-')
            plt.plot([0, epoch+1], [valid_accuracy_best]*2, 'r-')
            plt.ylabel('accuracy')
            plt.ylim([0.6, 1])

            plt.legend(['tacc', 'vacc'],
                       loc='upper left')
            plt.plot([epoch_best]*2, [0, 1], '-k')
            plt.grid()
            plt.show()
            plt.draw()

            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            print "model.theta.bias:", model.st.loc_net[-1].b.data

            print "theta:", theta.data[0]
            ax.matshow(cuda.to_cpu(x_batch[0]).reshape(in_shape),
                       cmap=plt.cm.gray)
            corners_x, corners_y = cuda.to_cpu(points.data[0])[
                :, [0, out_shape[1] - 1, -1, - out_shape[1]]]
#            print "theta:", theta_valid.data[0]
#            ax.matshow(x_valid_data[0].reshape(in_shape), cmap=plt.cm.gray)
#            corners_x, corners_y = points_valid.data[0][:, [0, 27, -1, -28]]
            ax.plot(corners_x[[0, 1]], corners_y[[0, 1]])
            ax.plot(corners_x[[1, 2]], corners_y[[1, 2]])
            ax.plot(corners_x[[2, 3]], corners_y[[2, 3]])
            ax.plot(corners_x[[0, 3]], corners_y[[0, 3]])
            ax.set_xlim([0, 60])
            ax.set_ylim([60, 0])

            ax = fig.add_subplot(1, 2, 2)
            ax.matshow(cuda.to_cpu(x_st.data[0]).reshape(out_shape),
                       cmap=plt.cm.gray)
#            ax.matshow(x_st_valid.data[0].reshape(out_shape), cmap=plt.cm.gray)
            plt.show()
            plt.draw()

    except KeyboardInterrupt:
        pass
