import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1/(1 + np.exp(-z))
    return g

pos = y == 1
neg = y == 0
pyplot.plot(X[pos, 0], X[pos, 1], 'k*', ms=10)
pyplot.plot(X[neg, 0], X[neg, 1], 'yo', ms=8, mec='k')
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'], loc='upper right')

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:, np.newaxis]
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:, np.newaxis]))
    return out

X = mapFeature(X[:, 0], X[:, 1])

m, n = X.shape
y = y[:, np.newaxis]

def costFunctionReg(theta, X, y, lambda_):
    m = y.size # m = 118
    n = theta.size # n = 28
    J = 0
    grad = np.zeros(theta.shape) # (n,1)
    h = sigmoid(np.dot(X, theta)) # (m,1)
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))) + (lambda_/(2*m)) * (np.dot(theta[1:].T, theta[1:]))
    grad[0] = (1/m) * (np.dot(X[:, 0].T, h-y))
    for i in range(1, n):
        grad[i] = (1/m) * (np.dot(X[:, i].T, h-y)) + (lambda_/m) * (theta[i])
    return J, grad # (1,1), (n,1)

initial_theta = np.zeros((n, 1)) # (n,1)

lambda_ = 1

cost, grad = costFunctionReg(initial_theta, X, y, lambda_)
print(grad.shape)

print('Cost at initial theta (zeros): ', cost.flatten())
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only: ', grad[:5].flatten())
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

test_theta = np.ones((n,1))
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    :', cost.flatten())
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('Gradient at initial theta (zeros) - first five values only: ', grad[:5].flatten())
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

initial_theta = np.zeros((n, 1))
lambda_ = 1
res = optimize.minimize(costFunctionReg,
                        initial_theta.flatten(),
                        (X, y.flatten(), lambda_),
                        jac=True,
                        method='TNC',
                        options={'maxiter': 3000})

cost = res.fun
theta = res.x
print("Optimized theta: ", theta)
print("Cost: ", cost)

def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    
    def mapFeaturePlot(X1, X2):
        degree = 6
        out = np.ones(1)
        for i in range(1, degree+1):
            for j in range(0, i+1):
                out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
        return out
    
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeaturePlot(u[i], v[j]), theta)
    
    pos = y.flatten() == 1
    neg = y.flatten() == 0
    X = data[:, 0:2]
    
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
       
    pyplot.contour(u,v,z,0)
    pyplot.title('Figure 4: Training data with decision boundary (λ = 1)\n', fontsize = 14)
    pyplot.xlabel('Microchip Test1')
    pyplot.ylabel('Microchip Test2')
    pyplot.legend(['y = 1', 'y = 0'], loc='upper right')

plotDecisionBoundary(theta, X, y)
pyplot.show()

def predict(theta, X):
    m = X.shape[0]
    p = np.zeros(m)
    pred = sigmoid(np.dot(X, theta))
    for i in range(m):
        if pred[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    
    return p

p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y.flatten()) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')