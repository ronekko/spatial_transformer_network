# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 01:34:50 2015

@author: ryuhei
"""

import unittest
import numpy as np
from chainer import Variable
from chainer import gradient_check
from spatial_transformer_network import (
    SpatialTransformer, GridGenerator, ImageSampler
)


class SpatialTransformerTest(unittest.TestCase):
    # TODO: Add test cases which some points are outside of the image region
    def test_spatial_transformer_call(self):
        # data
        x_data = np.arange(50, dtype=np.float32).reshape(2, 5, 5)

        # model
        in_shape = x_data.shape[1:]
        out_shape = (2, 2)
        spatial_transformer = SpatialTransformer(in_shape, out_shape)

        # forward and backward
        x = Variable(x_data)
        y, theta = spatial_transformer(x)
        y.grad = np.ones_like(y.data)
        y.backward(retain_grad=True)

        # y (data)
        expected_y = np.array([[0, 1, 5, 6],
                               [25, 26, 30, 31]], dtype=np.float32)
        assert np.allclose(y.data, expected_y)
        assert y.data.dtype == expected_y.dtype
        assert y.data.shape == expected_y.shape

        # theta (data)
        expected_theta = np.array([[0, 0],
                                   [0, 0]], dtype=np.float32)
        assert np.allclose(theta.data, expected_theta)
        assert theta.data.dtype == expected_theta.dtype
        assert theta.data.shape == expected_theta.shape

        # theta (gradient)
        expected_gtheta = np.array([[4, 20],
                                    [4, 20]], dtype=np.float32)
        assert np.allclose(theta.grad, expected_gtheta)
        assert theta.grad.dtype == expected_gtheta.dtype
        assert theta.grad.shape == expected_gtheta.shape

        # x (gradient)
        expected_gx = np.zeros_like(x_data)
        expected_gx[:, 0:2, 0:2] = 1
        assert np.allclose(x.grad, expected_gx)
        assert x.grad.dtype == expected_gx.dtype
        assert x.grad.shape == expected_gx.shape


    def test_spatial_transformer_parameteres(self):
        in_shape = (5, 5)
        out_shape = (2, 2)
        spatial_transformer = SpatialTransformer(in_shape, out_shape)
        assert len(spatial_transformer.parameters) != 0


class GridGeneratorTest(unittest.TestCase):
    def test_grid_genarator_forward(self):
        # data
        width, height = (4, 3)
        grid_shape = (width, height)
        theta = np.array([[10, 20], [300, 400], [5000, 6000]])

        # value calculated by GridGenerator
        grid_generator = GridGenerator(grid_shape)
        points_s = grid_generator(Variable(theta)).data

        # expected value
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points_t = np.vstack((x.ravel(), y.ravel()))
        expected_points_s = []
        for theta_i in theta:
            points_s_i = points_t + np.atleast_2d(theta_i).T
            expected_points_s.append(points_s_i)
        expected_points_s = np.array(expected_points_s).astype(np.float32)

        assert np.all(points_s == expected_points_s)
        assert points_s.dtype == expected_points_s.dtype
        assert points_s.shape == expected_points_s.shape


class ImageSamplerTest(unittest.TestCase):
    # TODO: Add test cases which some points are outside of the image region
    def test_image_sampler_forward(self):
        # data
        x = np.arange(50, dtype=np.float32).reshape(2, 5, 5)
        points = np.array([[[0, 1, 2, 3, 4],
                            [3, 2, 1, 0, 4]],
                           [[0.1, 1.3, 2.5, 3.7, 3.99],
                            [0.1, 1.3, 2.5, 3, 3.99]]], dtype=np.float32)

        # a
        image_sampler = ImageSampler()
        y = image_sampler(Variable(x), Variable(points)).data
        print y

        # expected value
        y0 = [15.0, 11.0, 7.0, 3.0, 24.0]
        y1 = []
        y1.append(0.9 * (0.9 * 25 + 0.1 * 26) +
                  0.1 * (0.9 * 30 + 0.1 * 31))
        y1.append(0.7 * (0.7 * 31 + 0.3 * 32) +
                  0.3 * (0.7 * 36 + 0.3 * 37))
        y1.append(0.5 * (0.5 * 37 + 0.5 * 38) +
                  0.5 * (0.5 * 42 + 0.5 * 43))
        y1.append(1.0 * (0.3 * 43 + 0.7 * 44) +
                  0.0 * (0.3 * 48 + 0.7 * 49))
        y1.append(0.01 * (0.01 * 43 + 0.99 * 44) +
                  0.99 * (0.01 * 48 + 0.99 * 49))
        expected_y = np.array([y0, y1], dtype=np.float32)
        print expected_y

        assert np.allclose(y, expected_y)
        assert y.dtype == expected_y.dtype
        assert y.shape == expected_y.shape


    def test_image_sampler_backward(self):
        x_data = np.arange(50, dtype=np.float32).reshape(2, 5, 5)
        points_data = np.array([[[1, 1, 2, 3],
                                [3, 2, 1, 1]],
                               [[0.2, 1.3, 2.5, 3.7],
                                [0.2, 1.3, 2.5, 3]]], dtype=np.float32)

        x = Variable(x_data)
        points = Variable(points_data)
        image_sampler = ImageSampler()
        y = image_sampler(x, points)

        func = lambda: image_sampler.forward((x_data, points_data))
        y.grad = np.ones_like(y.data)
        y.backward()
        grad = gradient_check.numerical_grad(func,
                                             (x_data, points_data),
                                             (y.grad,),
                                             eps=1e-1)
        gx_numerical, gpoints_numerical = grad

        print "Check gx:"
        print "gx:", x.grad
        print "gx (numerical):", gx_numerical
        gradient_check.assert_allclose(x.grad, gx_numerical)
        print "Check gpoints:"
        print "gpoints:", points.grad
        print "gpoints (numerical):", gpoints_numerical
        gradient_check.assert_allclose(points.grad, gpoints_numerical)


    def test_image_sampler_backward2(self):
        # data
        x_data = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
        x_data = np.vstack((x_data, x_data, x_data))

        points_data = []
        x0 = np.array([0.5, 2.0, 3.5])
        y0 = np.array([0.5, 2.0, 3.5])
        x, y = np.meshgrid(x0, y0)
        x = x.ravel()
        y = y.ravel()
        points_data.append([x, y])

        x1 = np.array([0, 2, 4])
        y1 = np.array([0, 2, 4])
        x, y = np.meshgrid(x1, y1)
        x = x.ravel()
        y = y.ravel()
        points_data.append([x, y])

        x2 = np.array([-2, -1, -0.1])
        y2 = np.array([4.1, 5, 6])
        x, y = np.meshgrid(x2, y2)
        x = x.ravel()
        y = y.ravel()
        points_data.append([x, y])
        points_data = np.array(points_data, dtype=np.float32)
        assert points_data.shape == (3, 2, 9)
        assert points_data.dtype == np.float32
        print points_data

        # model
        image_sampler = ImageSampler()

        # forward and backward
        x = Variable(x_data)
        points = Variable(points_data)
        y = image_sampler(x, points)
        y.grad = np.ones_like(y.data)
        y.backward()

        # y (data)
        expected_y = np.array([[3, 4.5, 6, 10.5, 12, 13.5, 18, 19.5, 21],
                               [0, 2, 4, 10, 12, 14, 20, 22, 24],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              dtype=np.float32)
        gradient_check.assert_allclose(y.data, expected_y)
        assert y.data.dtype == expected_y.dtype
        assert y.data.shape == expected_y.shape

        func = lambda: image_sampler.forward((x_data, points_data))
        y.grad = np.ones_like(y.data)
        y.backward()
        grad = gradient_check.numerical_grad(func,
                                             (x_data, points_data),
                                             (y.grad,),
                                             eps=1e-0)
        gx_numerical, gpoints_numerical = grad

        # check gx
        print "Check gx:"
        print "gx:", x.grad
        print "gx (numerical):", gx_numerical
        gradient_check.assert_allclose(x.grad, gx_numerical)

        # check gpoints (Note that the numerical gradient is undefiend where
        # points near or out of boudary.)
        expected_gpoints = np.array([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 5, 5, 5, 5, 5, 5, 5, 5]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],
             [5, 5, 5, 5, 5, 5, 5, 5, 5]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.float32)
        print "Check gpoints:"
        print "gpoints:", points.grad
        print "gpoints (numerical):", gpoints_numerical
        gradient_check.assert_allclose(points.grad, expected_gpoints)


if __name__ == '__main__':
    unittest.main()
