# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 22:33:53 2015

@author: ryuhei
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import chainer.functions as F
from chainer import Variable, FunctionSet, Function
from chainer import cuda, gradient_check


def load_cluttered_mnist():
    FILE_NAME = "mnist_cluttered_60x60_6distortions.npz"
    DATA_URL = "https://s3.amazonaws.com/lasagne/recipes/datasets/" + FILE_NAME
    if not os.path.exists(FILE_NAME):
        subprocess.check_call(['wget', '-N', DATA_URL])

    dataset = np.load(FILE_NAME)
    x_train, x_valid, x_test = [dataset[name].reshape(-1, 60, 60)
                                for name in ["x_train", "x_valid", "x_test"]]
    y_train, y_valid, y_test = [dataset[name].argmax(axis=1).astype(np.int32)
                                for name in ["y_train", "y_valid", "y_test"]]
    return (x_train, y_train, x_valid, y_valid, x_test, y_test)


# TODO: If a sampling point is just on a grid intersection, just use the pixel
# value at the point. (no interpolation is needed because the weights for
# neighboring pixels are 0)
class ImageSampler(Function):
    def forward(self, inputs):
        U, points = inputs
        batch_size, height, width = U.shape

        batch_axis = np.expand_dims(np.arange(batch_size), 1)
        points_floor = np.floor(points)
        x_l = points_floor[:, 0].astype(np.int32)
        y_l = points_floor[:, 1].astype(np.int32)
        x_l = np.clip(x_l, 0, width - 1)
        y_l = np.clip(y_l, 0, height - 1)
        x_h = np.clip(x_l + 1, 0, width - 1)
        y_h = np.clip(y_l + 1, 0, height - 1)

        weight = 1.0 - (points - points_floor)
        weight_x_l = weight[:, 0]
        weight_y_l = weight[:, 1]
        weight_x_h = 1 - weight_x_l
        weight_y_h = 1 - weight_y_l

        # remove points outside of the (source) image region
        # by setting their weights to 0
        x = points[:, 0]
        y = points[:, 1]
        x_invalid = np.logical_or(x < 0, (width - 1) < x)
        y_invalid = np.logical_or(y < 0, (height - 1) < y)
        invalid = np.logical_or(x_invalid, y_invalid)
        weight_x_l[invalid] = 0
        weight_y_l[invalid] = 0
        weight_x_h[invalid] = 0
        weight_y_h[invalid] = 0

        U_y_l = (weight_x_l * U[batch_axis, y_l, x_l] +
                 weight_x_h * U[batch_axis, y_l, x_h])
        U_y_h = (weight_x_l * U[batch_axis, y_h, x_l] +
                 weight_x_h * U[batch_axis, y_h, x_h])
        V = weight_y_l * U_y_l + weight_y_h * U_y_h

        self.x_l = x_l
        self.y_l = y_l
        self.x_h = x_h
        self.y_h = y_h
        self.weight_x_l = weight_x_l
        self.weight_y_l = weight_y_l
        self.weight_x_h = weight_x_h
        self.weight_y_h = weight_y_h
        return (V,)

    def backward(self, inputs, grad_outputs):
        U, points = inputs
        gV, = grad_outputs

        x_l = self.x_l
        y_l = self.y_l
        x_h = self.x_h
        y_h = self.y_h
        weight_x_l = self.weight_x_l
        weight_y_l = self.weight_y_l
        weight_x_h = self.weight_x_h
        weight_y_h = self.weight_y_h

        batch_size, height, width = U.shape
        dims = height * width

        # gU ############################################################
        gU = np.empty_like(U)
        gU_flat = gU.reshape(batch_size, dims)

        i_ll = width * y_l + x_l  # i_ll.shape = (batch_size, dims)
        i_lh = width * y_l + x_h
        i_hl = width * y_h + x_l
        i_hh = width * y_h + x_h
        i = np.hstack((i_ll, i_lh, i_hl, i_hh))
        w_ll = weight_y_l * weight_x_l * gV
        w_lh = weight_y_l * weight_x_h * gV
        w_hl = weight_y_h * weight_x_l * gV
        w_hh = weight_y_h * weight_x_h * gV
        w = np.hstack((w_ll, w_lh, w_hl, w_hh))

        for b in range(batch_size):
            gU_flat[b] = np.bincount(i[b], weights=w[b], minlength=dims)
        # gU ############################################################

        # gpoints #######################################################
        batch_axis = np.expand_dims(np.arange(batch_size), 1)
        U_ll = U[batch_axis, y_l, x_l]
        U_lh = U[batch_axis, y_l, x_h]
        U_hl = U[batch_axis, y_h, x_l]
        U_hh = U[batch_axis, y_h, x_h]

        # y == weight_y_h, (1 - y) == weight_y_l
        gx = weight_y_l * (U_lh - U_ll) + weight_y_h * (U_hh - U_hl)
        gy = weight_x_l * (U_hl - U_ll) + weight_x_h * (U_hh - U_lh)

        gx = gx * gV
        gy = gy * gV

        # shape o gx, gy: (N, P)
        gx = np.expand_dims(gx, 1)
        gy = np.expand_dims(gy, 1)
        gpoints = np.hstack((gx, gy))
        # gpoints #######################################################

        return (gU, gpoints)


class GridGenerator(Function):
    def __init__(self, grid_size):
        """
        Args:
            grid_size (tuple): Shape (width, height) of target image.
        """
        self.grid_size = grid_size
        width, height = grid_size
        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        one = np.ones(grid_size)
        # G in 3.2 "Parameterised Sampling Grid
        self.points_t = np.vstack((x.ravel(),
                                   y.ravel(),
                                   one.ravel())).astype(np.float32)

    def forward(self, inputs):
        theta, = inputs
        batch_size = len(theta)
        width, height = self.grid_size

        eyes_3d = np.repeat(np.expand_dims(np.eye(2), 0), batch_size, axis=0)
        theta_3d = np.expand_dims(theta, 2)
        A = np.dstack((eyes_3d, theta_3d))  # transformation matrix
        points_s = np.dot(A, self.points_t).astype(np.float32)
        return (points_s,)

    def backward(self, inputs, grad_outputs):
        gpoints, = grad_outputs
        gtheta = gpoints.sum(axis=2)
        return (gtheta,)


class FCLocalizationNetwork(FunctionSet):
    def __init__(self, in_size, theta_size):
        sqrt2 = np.sqrt(2)
        super(FCLocalizationNetwork, self).__init__(
            fc1=F.Linear(in_size, 32, wscale=1),
            fc2=F.Linear(32, 32, wscale=sqrt2),
            fc3=F.Linear(32, 32, wscale=sqrt2),
            fc4=F.Linear(32, theta_size,
                         initialW=np.zeros((theta_size, 32), dtype=np.float32))
        )

    def __call__(self, x, train=False):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        theta = self.fc4(h)  # theta has the shape of (len(x), theta_size)
        return theta


class SpatialTransformer(Function):
    """See A.3 of "Spatial Transformer Networks"
       http://arxiv.org/abs/1506.02025v1
    """
    def __init__(self, in_shape, out_shape, loc_net=None):
        self.in_shape = in_shape  # (height, width)
        self.out_shape = out_shape
        self.theta_size = 2  # number of parameters for translation
        in_size = in_shape[0] * in_shape[1]
        if loc_net is None:
            self.loc_net = FCLocalizationNetwork(in_size, self.theta_size)
        else:
            self.loc_net = loc_net
        self.grid_generator = GridGenerator(out_shape)
        self.image_sampler = ImageSampler()

    def __call__(self, x):
        theta = self.loc_net(x)  # theta has the shape of (len(x), theta_size)
        points_s = self.grid_generator(theta)
        y = self.image_sampler(x, points_s)
        return (y, theta)

    @property
    def parameters(self):
        return self.loc_net.parameters

    @parameters.setter
    def parameters(self, params):
        self.loc_net.parameters = params

    @property
    def gradients(self):
        return self.loc_net.gradients

    @gradients.setter
    def gradients(self, grads):
        self.loc_net.gradients = grads

if __name__ == '__main__':
    try:
        x_train
    except NameError:
        cluttered_mnist = load_cluttered_mnist()
        x_train, y_train, x_valid, y_valid, x_test, y_test = cluttered_mnist

    plt.matshow(x_train[0], cmap=plt.cm.gray)
    plt.show()

#    # test 1 ###########################################
#    # data
#    x_data = np.arange(50, dtype=np.float32).reshape(2, 5, 5)  # input images
#    height, width = (2, 2)  # target shape
#    grid_shape = (height, width)
#    points_data = np.array([[[0, 1, 2, 3, 4],
#                            [4, 3, 2, 1, 0]],
#                           [[0.1, 1.3, 2.5, 3.7, 3.99],
#                            [0.1, 1.3, 2.5, 3, 3.99]]], dtype=np.float32)
#    theta_data = np.array([[1, 1], [2.5, 2.5]], dtype=np.float32)
#    x = Variable(x_data)
#    theta = Variable(theta_data)
#    print "x"
#    print x_data
#    print "theta"
#    print theta_data
#
#    grid_generator = GridGenerator(grid_shape)
#    points_s = grid_generator(theta)
#    image_sampler = ImageSampler()
#    y = image_sampler(x, points_s)
#    print "y"
#    print y.data.reshape((-1,) + grid_shape)
#    # test 1 ###########################################

#    # test 2 ###########################################
#    # Clip small image by GridGenerator and ImageSampler
#    height, width = (28, 28)  # target shape
#    grid_shape = (height, width)
#    grid_generator = GridGenerator(grid_shape)
#    theta_data = np.array([[20, 5],
#                           [20.5, 5.5],
#                           [23, 8],
#                           [26, 11],
#                           [29, 14]], dtype=np.float32)
#    x = Variable(x_train[[0]*len(theta_data)])
#    theta = Variable(theta_data)
#    points_s = grid_generator(theta)
#    image_sampler = ImageSampler()
#    y = image_sampler(x, points_s)
#
#    for y_i in y.data:
#        plt.matshow(y_i.reshape(28, 28), cmap=plt.cm.gray)
#    plt.show()
#    # test 2 ###########################################

#    # test 3 ###########################################
#    theta_data = np.array([[-6.2, -5.6], [40, 42]], dtype=np.float32)
#    x_data = x_train[[0]*len(theta_data)]
#    theta = Variable(theta_data)
#    x = Variable(x_data)
#
#    height, width = (28, 28)  # target shape
#    grid_shape = (height, width)
#    grid_generator = GridGenerator(grid_shape)
#
#    points_s = grid_generator(theta)
#    points_s_data = points_s.data
#    image_sampler = ImageSampler()
#    y = image_sampler(x, points_s)
#
#    for y_i in y.data:
#        plt.matshow(y_i.reshape(28, 28), cmap=plt.cm.gray)
#    plt.show()
#
#    func = lambda: image_sampler.forward((x_data, points_s_data))
#    y.grad = np.full(y.data.shape, 1, dtype=np.float32)
#    y.backward()
#    grad = gradient_check.numerical_grad(func,
#                                         (x_data, points_s_data),
#                                         (y.grad,),
#                                         1e-2)
#    gx_numerical, gpoints_s_numerical = grad
#
#    plt.matshow(gx_numerical[0].reshape(60, 60), cmap=plt.cm.gray)
#    plt.matshow(x.grad[0].reshape(60, 60), cmap=plt.cm.gray)
#    plt.show()
#
#    print "gx:", x.grad
#    print "gx (numerical):", gx_numerical
#    gradient_check.assert_allclose(x.grad, gx_numerical)
#    gradient_check.assert_allclose(points_s.grad, gpoints_s_numerical)
#    # test 3 ###########################################

    # test 4 ###########################################
    # SpatialTransformer ###############################
    x_data = x_train[:2]
    in_shape = x_data.shape[1:]
    out_shape = (28, 28)
    x = Variable(x_data)
    spatial_transformer = SpatialTransformer(in_shape, out_shape)
    y, theta = spatial_transformer(x)
    print theta.data
    y_data = y.data

    plt.matshow(y_data[0].reshape(out_shape), cmap=plt.cm.gray)
    plt.matshow(y_data[1].reshape(out_shape), cmap=plt.cm.gray)
    y.grad = np.ones_like(y_data)
    y.backward()

    # show x.grad
    plt.matshow(x.grad[0].reshape(in_shape), cmap=plt.cm.gray)
    plt.matshow(x.grad[1].reshape(in_shape), cmap=plt.cm.gray)
    # test 4 ###########################################
