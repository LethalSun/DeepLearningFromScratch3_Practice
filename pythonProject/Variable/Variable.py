import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not ndarray.'.format((type(data))))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.in_var, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, in_var):
        x = in_var.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.in_var = in_var
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


def square(x):
    return Square()(x)


def exp(x):
    return  Exp()(x)


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
