import unittest
import Variable
import numpy as np


def numerical_diff(f, x, eps = 1e-4):
    h_x = Variable.Variable(x.data - eps)
    x_h = Variable.Variable(x.data + eps)

    h_y = f(h_x)
    y_h = f(x_h)

    h2 = 2 * eps

    return (y_h.data - h_y.data) / h2


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable.Variable(np.array(2.0))
        y = Variable.square(x)
        expected = np.array(4.0)

        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable.Variable(np.array(3.0))
        y = Variable.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable.Variable(np.random.rand(1))
        y = Variable.square(x)
        y.backward()

        num_grad = numerical_diff(Variable.square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)