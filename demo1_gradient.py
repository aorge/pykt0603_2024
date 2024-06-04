from matplotlib import pyplot as plt
import numpy
from sklearn import datasets

# from numpy import abs

regressionData = datasets.make_regression(100, 1, noise=5)
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
init_m = 10
init_b = 10
learning_rate = 0.1
range1 = [-5, 5]
plt.plot(range1, init_m * numpy.array(range1) + init_b)


def cost(m, b, X, Y):
    N = len(X)
    cost = 0
    for i in range(N):
        cost += (Y[i] - (m * X[i] + b)) ** 2
    return cost


init_cost = cost(init_m, init_b, regressionData[0], regressionData[1])
print("一開始的誤差是:{}".format(init_cost))
plt.show()


def update_weights(m, b, X, Y, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        m_deriv += -2 * X[i] * (Y[i] - (m * X[i] + b))
        b_deriv += -2 * (Y[i] - (m * X[i] + b))
    m -= learning_rate * (m_deriv / len(X))
    b -= learning_rate * (b_deriv / len(X))
    return m, b


current_m = init_m
current_b = init_b


for _ in range(50):
    new_m, new_b = update_weights(current_m, current_b, regressionData[0], regressionData[1], learning_rate)
    print("新的m={}, b={}".format(new_m, new_b))
    new_cost = cost(new_m, new_b, regressionData[0], regressionData[1])
    print("cost=", new_cost)
    plt.plot(range1, new_m * range1 + new_b)
    plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
    plt.title("cost={}".format(new_cost))
    plt.show()
    current_m = new_m
    current_b = new_b