import numpy as np


class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, in_var):
        x = in_var.data
        y = self.forward(x)
        output = Variable(y)
        self.in_var = in_var
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.in_var.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.in_var.data
        gx = np.exp(x) * gy
        return gx
