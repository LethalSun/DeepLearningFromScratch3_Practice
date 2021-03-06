import numpy as np
import Variable

import sys

def numerical_diff(f, x, eps = 1e-4):
    h_x = Variable.Variable(x.data - eps)
    x_h = Variable.Variable(x.data + eps)

    h_y = f(h_x)
    y_h = f(x_h)

    h2 = 2 * eps

    return (y_h.data - h_y.data) / h2

if __name__ == '__main__':
    # data = np.array(1.0)
    # x = Variable.Variable(data)
    # print(x.data)
    #
    # x.data = np.array(2.0)
    # print(x.data)
    #
    # x = np.array(1)
    # print(x.ndim)
    # x = np.array([1, 2, 3])
    # print(x.ndim)
    # x = np.array(([[1, 2, 3],[4, 5, 6]]))
    # print(x.ndim)
    #
    # x = Variable.Variable(np.array(10))
    # f = Variable.Square()
    # y = f(x)
    # print(type(y))
    # print(y.data)

    # x = Variable.Variable(np.array(0.5))
    #
    # sq = Variable.Square()
    # exp = Variable.Exp()
    #
    # a = sq(x)
    # b = exp(a)
    # c = sq(b)
    #
    # print(type(c))
    # print(c.data)

    # f = Variable.Square()
    # x = Variable.Variable(np.array(2.0))
    #
    # dy = numerical_diff(f, x)
    #
    # print(dy)

    # def ff(x):
    #     A = Variable.Square()
    #     B = Variable.Exp()
    #     C = Variable.Square()
    #
    #     return C(B(A(x)))
    #
    # x = Variable.Variable(np.array(0.5))
    # dy = numerical_diff(ff, x)
    #
    # print(dy)
    #
    #
    # def fff(x):
    #     A = Variable.Square()
    #     B = Variable.Exp()
    #
    #     return A(B(A(x)))
    #
    #
    # x = Variable.Variable(np.array(0.5))
    # dy = numerical_diff(fff, x)
    #
    # print(dy)

    A = Variable.Square()
    B = Variable.Exp()
    C = Variable.Square()

    x = Variable.Variable(np.array(0.5))

    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.in_var == b
    assert y.creator.in_var.creator == B
    assert y.creator.in_var.creator.in_var == a
    assert y.creator.in_var.creator.in_var.creator == A
    assert y.creator.in_var.creator.in_var.creator.in_var == x

    y.grad = np.array(1.0)

    y.backward()

    # C = y.creator
    # b = C.in_var
    # b.grad = C.backward(y.grad)
    #
    # B = b.creator
    # a = B.in_var
    # a.grad = B.backward(b.grad)
    #
    # A = a.creator
    # x = A.in_var
    # x.grad = A.backward(a.grad)

    # b.grad = C.backward(y.grad)
    # a.grad = B.backward(b.grad)
    # x.grad = A.backward(a.grad)
    print(x.grad)
    print(x.grad)

    x = Variable.Variable(np.array(0.5))
    a = Variable.square(x)
    b = Variable.exp(a)
    y = Variable.square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    x = Variable.Variable(np.array(0.5))
    y = Variable.square(Variable.exp(Variable.square(x)))
    y.backward()
    print(x.grad)

    x = Variable.Variable(np.array(0.5))