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
from chainer.utils import type_check


# TODO: write test
_place_kernel = cuda.cupy.ElementwiseKernel(
    'T a, S mask, T val',
    'T out',
    'out = mask ? val : a',
    'cupy_place')


# TODO: write test
def cupy_place(arr, mask, val):
    val = cuda.cupy.array(val, dtype=arr.dtype)
    out = cuda.cupy.empty_like(arr)
    return _place_kernel(arr, mask, val, out)


# TODO: write test
def apply_with_mask(arr, mask, elementwise_func, *args):
    value = elementwise_func(arr, *args)
    return cupy_place(arr, mask, value)


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


class ImageSampler(Function):
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        U_type, points_type = in_types

        type_check.expect(
            U_type.dtype == np.float32,
            U_type.ndim == 3
        )
        type_check.expect(
            points_type.dtype == np.float32,
            points_type.ndim == 3,
            points_type.shape[0] == U_type.shape[0],
            points_type.shape[1] == 2
        )

    def forward_cpu(self, inputs):
        U, points = inputs
        batch_size, height, width = U.shape

        # Points just on the boundary are slightly (i.e. nextafter in float32)
        # moved inward to simplify the implementation
        points = points.copy()
        on_boundary = (points == 0)
        points[on_boundary] = np.nextafter(points[on_boundary], np.float32(1))
        x = points[:, 0]
        y = points[:, 1]
        on_boundary = (x == (width - 1))
        x[on_boundary] = np.nextafter(x[on_boundary], np.float32(0))
        on_boundary = (y == (height - 1))
        y[on_boundary] = np.nextafter(y[on_boundary], np.float32(0))

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

    def forward_gpu(self, inputs):
        xp = cuda.get_array_module(*inputs)
        U, points = inputs
        batch_size, height, width = U.shape

        # Points just on the boundary are slightly (i.e. nextafter in float32)
        # moved inward to simplify the implementation
        points = points.copy()
        on_boundary = (points == 0)
        ret = apply_with_mask(points, on_boundary, xp.nextafter, xp.float32(1))
        points = cupy_place(points, on_boundary, ret)
        x = points[:, 0]
        y = points[:, 1]
        on_boundary = (x == (width - 1))
        ret = apply_with_mask(x, on_boundary, xp.nextafter, xp.float32(0))
        x = cupy_place(x, on_boundary, ret)
        on_boundary = (y == (height - 1))
        ret = apply_with_mask(y, on_boundary, xp.nextafter, xp.float32(0))
        y = cupy_place(y, on_boundary, ret)

        points_floor = xp.floor(points)
        x_l = points_floor[:, 0].astype(xp.int32)
        y_l = points_floor[:, 1].astype(xp.int32)
        x_l = xp.clip(x_l, 0, width - 1)
        y_l = xp.clip(y_l, 0, height - 1)
        x_h = xp.clip(x_l + 1, 0, width - 1)
        y_h = xp.clip(y_l + 1, 0, height - 1)

        weight = xp.float32(1.0) - (points - points_floor)
        weight_x_l = weight[:, 0]
        weight_y_l = weight[:, 1]
        weight_x_h = xp.float32(1.0) - weight_x_l
        weight_y_h = xp.float32(1.0) - weight_y_l

        # remove points outside of the (source) image region
        # by setting their weights to 0
        x_invalid = xp.logical_or(x < 0, (width - 1) < x)
        y_invalid = xp.logical_or(y < 0, (height - 1) < y)
        invalid = xp.logical_or(x_invalid, y_invalid)
        weight_x_l = cupy_place(weight_x_l, invalid, 0)
        weight_y_l = cupy_place(weight_y_l, invalid, 0)
        weight_x_h = cupy_place(weight_x_h, invalid, 0)
        weight_y_h = cupy_place(weight_y_h, invalid, 0)

        size_U = width * height
        size_V = points.shape[-1]
        batch_index = xp.asarray(
            np.repeat(np.arange(batch_size, dtype=np.int32), size_V))
        index_ll = batch_index * size_U + y_l.ravel() * width + x_l.ravel()
        self.U_ll = xp.take(U, index_ll).reshape((batch_size, -1))
        index_hl = batch_index * size_U + y_l.ravel() * width + x_h.ravel()
        self.U_lh = xp.take(U, index_hl).reshape((batch_size, -1))
        index_lh = batch_index * size_U + y_h.ravel() * width + x_l.ravel()
        self.U_hl = xp.take(U, index_lh).reshape((batch_size, -1))
        index_hh = batch_index * size_U + y_h.ravel() * width + x_h.ravel()
        self.U_hh = xp.take(U, index_hh).reshape((batch_size, -1))

        U_y_l = weight_x_l * self.U_ll + weight_x_h * self.U_lh
        U_y_h = weight_x_l * self.U_hl + weight_x_h * self.U_hh
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

    def backward_cpu(self, inputs, grad_outputs):
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

    def backward_gpu(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
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
        i_ll = width * y_l + x_l  # i_ll.shape = (batch_size, dims)
        i_lh = width * y_l + x_h
        i_hl = width * y_h + x_l
        i_hh = width * y_h + x_h
        i = xp.hstack((i_ll, i_lh, i_hl, i_hh)).get()
        w_ll = weight_y_l * weight_x_l * gV
        w_lh = weight_y_l * weight_x_h * gV
        w_hl = weight_y_h * weight_x_l * gV
        w_hh = weight_y_h * weight_x_h * gV
        w = xp.hstack((w_ll, w_lh, w_hl, w_hh)).get()

        # compute gU with np.bincount on CPU then bring back it to GPU again
        gU_flat = np.empty((batch_size, dims), dtype=np.float32)
        for b in range(batch_size):
            gU_flat[b] = np.bincount(i[b], weights=w[b], minlength=dims)
        gU = xp.asarray(gU_flat.reshape(U.shape))
        # gU ############################################################

        # gpoints #######################################################
        U_ll = self.U_ll
        U_lh = self.U_lh
        U_hl = self.U_hl
        U_hh = self.U_hh

        # y == weight_y_h, (1 - y) == weight_y_l
        gx = weight_y_l * (U_lh - U_ll) + weight_y_h * (U_hh - U_hl)
        gy = weight_x_l * (U_hl - U_ll) + weight_x_h * (U_hh - U_lh)

        gx = gx * gV
        gy = gy * gV

        # shape o gx, gy: (N, P)
        gx = xp.expand_dims(gx, 1)
        gy = xp.expand_dims(gy, 1)
        gpoints = xp.hstack((gx, gy))
        # gpoints #######################################################

        return (gU, gpoints)


class GridGeneratorTranslation(Function):
    def __init__(self, in_shape, out_shape):
        """
        Args:
           in_shape (tuple): (height, width) of source image.

           out_shape (tuple): (height, width) of target image.
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        out_height, out_width = out_shape

        x = np.arange(out_width) - out_width / 2.0
        y = np.arange(out_height) - out_height / 2.0
        x, y = np.meshgrid(x, y, indexing='xy')
        one = np.ones(out_shape)
        # G in 3.2 "Parameterised Sampling Grid
        self.points_t = np.vstack((x.ravel(),
                                   y.ravel(),
                                   one.ravel())).astype(np.float32)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        theta_type, = in_types

        type_check.expect(
            theta_type.dtype == np.float32,
            theta_type.ndim == 2,
            theta_type.shape[1] == 2
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        theta, = inputs
        batch_size = len(theta)
        in_height, in_width = self.in_shape

        # use np.tile() because cupy.tile() has not been implemented yet
        eyes_3d = xp.asarray(np.tile(np.eye(2), (batch_size, 1, 1)))
        theta_3d = xp.expand_dims(theta, 2)
        A = xp.dstack((eyes_3d, theta_3d))  # transformation matrix
        points_s = xp.dot(A, self.points_t).astype(xp.float32)

        offset = xp.array([in_width / 2.0, in_height / 2.0], dtype=xp.float32)
        offset = offset.reshape(1, -1, 1)
        points_s += offset
        return (points_s,)

    def backward(self, inputs, grad_outputs):
        gpoints, = grad_outputs
        gtheta = gpoints.sum(axis=2)
        return (gtheta,)


class GridGeneratorAffine(Function):
    def __init__(self, in_shape, out_shape):
        """
        Args:
           in_shape (tuple): (height, width) of source image.

           out_shape (tuple): (height, width) of target image.
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        out_height, out_width = out_shape

        x = np.arange(out_width) - (out_width - 1.0) / 2.0
        y = np.arange(out_height) - (out_height - 1.0) / 2.0
        x, y = np.meshgrid(x, y, indexing='xy')
        one = np.ones(out_shape) * (out_width - 1.0) / 2.0
        # G in 3.2 "Parameterised Sampling Grid
        self.points_t = np.vstack((x.ravel(),
                                   y.ravel(),
                                   one.ravel())).astype(np.float32)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        theta_type, = in_types

        type_check.expect(
            theta_type.dtype == np.float32,
            theta_type.ndim == 2,
            theta_type.shape[1] == 6
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        theta, = inputs
        batch_size = len(theta)
        in_height, in_width = self.in_shape

        A = theta.reshape(batch_size, 2, 3)
        points_s = xp.dot(A, self.points_t).astype(xp.float32)

        offset = xp.array([(in_height - 1.0) / 2.0,
                           (in_width - 1.0) / 2.0], dtype=xp.float32)
        offset = offset.reshape(1, -1, 1)
        points_s += offset
        return (points_s,)

    def backward(self, inputs, grad_outputs):
        gpoints, = grad_outputs  # shape: (b, 2, h*w)
        batch_size = len(gpoints)
        gtheta = gpoints.dot(self.points_t.T)
        gtheta = gtheta.reshape(batch_size, 6)
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

       Args:
           in_shape (tuple): (height, width) of source image.

           out_shape (tuple): (height, width) of target image.
    """
    def __init__(self, in_shape, out_shape, transformation="translation",
                 loc_net=None):
        assert len(in_shape) == 2, "in_shape must be (height, width)"
        assert len(out_shape) == 2, "out_shape must be (height, width)"
        self.in_shape = in_shape  # (height, width)
        self.out_shape = out_shape  # (height, width)

        # set number of parameters for transformation according to its type
        if transformation == "translation":
            self.theta_size = 2
            self.grid_generator = GridGeneratorTranslation(in_shape, out_shape)
        if transformation == "affine":
            self.theta_size = 6
            self.grid_generator = GridGeneratorAffine(in_shape, out_shape)

        in_size = np.prod(in_shape)
        if loc_net is None:
            self.loc_net = FCLocalizationNetwork(in_size, self.theta_size)
        else:
            self.loc_net = loc_net

        # set initial bias of the last layer of localization network
        theta_bias_init = self.loc_net.parameters[-1]
        translation_init = np.zeros(2, dtype=np.float32)
        if transformation == "translation":
            theta_bias_init[:] = translation_init
        if transformation == "affine":
            theta_bias_init[[0, 4]] = 1
            theta_bias_init[[2, 5]] = translation_init

        self.image_sampler = ImageSampler()

    # forward of SpatialTransformer is a series of forward operations
    # of existing functions. So we do not need additional type checks.
    def check_type_forward(self, in_types):
        pass

    def __call__(self, x, return_points=False):
        theta = self.loc_net(x)  # theta has the shape of (len(x), theta_size)
        points_s = self.grid_generator(theta)
        y = self.image_sampler(x, points_s)
        if return_points:
            return (y, theta, points_s)
        else:
            return (y, theta)

    def to_gpu(self, device=None):
        self.loc_net.to_gpu(device)
        self.grid_generator.to_gpu(device)
        self.image_sampler.to_gpu(device)
        return self

    def to_cpu(self):
        self.loc_net.to_cpu()
        self.grid_generator.to_cpu()
        self.image_sampler.to_cpu()
        return self

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
    spatial_transformer = SpatialTransformer(in_shape, out_shape, "affine")
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
