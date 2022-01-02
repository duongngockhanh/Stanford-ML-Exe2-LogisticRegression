import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1/(1 + np.exp(-z))
    return g

#plotData(X, y)
pos = y == 1
neg = y == 0

pyplot.plot(X[pos, 0], X[pos, 1], 'k*', ms=10)
pyplot.plot(X[neg, 0], X[neg, 1], 'yo', ms=8, mec='k')

# Labels and Legend
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')

# Specified in plot order
pyplot.legend(['y = 1', 'y = 0'], loc='upper right')
# pyplot.show()

def mapFeature(X1, X2):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    degree = 6
    out = np.ones(X.shape[0])[:, np.newaxis]
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:, np.newaxis]))
    return out

# Note that mapFeature also adds a column of ones for us, so the intercept term is included
X = mapFeature(X[:, 0], X[:, 1])

# Setup the data matrix appropriately
m, n = X.shape
y = y[:, np.newaxis] # Convert label (y) from 1D array to 2D array of shape (m, 1)

def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total 
        number of polynomial features. 
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).
    
    y : array_like
        The data labels. A vector with shape (m, ).
    
    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the regularized cost function. 
    
    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    
    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    n = theta.size

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ===================== YOUR CODE HERE ======================

    # Compute h
    h = sigmoid(np.dot(X, theta))
    
    # Compute J
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))) + (lambda_/(2*m)) * (np.dot(theta[1:].T, theta[1:]))
    
    # Compute grad for j = 0
    grad[0] = (1/m) * (np.dot(X[:, 0].T, h-y))
    
    # Compute grad for j >= 1
    for i in range(1, n):
        grad[i] = (1/m) * (np.dot(X[:, i].T, h-y)) + (lambda_/m) * (theta[i])
    
    # =============================================================
    return J, grad

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): ', cost.flatten())
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only: ', grad[:5].flatten())
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((n,1))
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    :', cost.flatten())
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('Gradient at initial theta (zeros) - first five values only: ', grad[:5].flatten())
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1
res = optimize.minimize(costFunctionReg,
                        initial_theta.flatten(),
                        (X, y.flatten(), lambda_),
                        jac=True,
                        method='TNC',
                        options={'maxiter': 3000})

# The fun property of OptimizeResult object returns the value of costFunctionReg at optimized theta
cost = res.fun

# The optimized theta is in the x property of the result
theta = res.x

print("Optimized theta: ", theta)
print("Cost: ", cost)

def plotDecisionBoundary(theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with + for the positive examples and o for  the negative examples.

    Parameters
    ----------

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, 1).
    """
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
    
    # Plot Examples
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
       
    pyplot.contour(u,v,z,0)
    pyplot.title('Figure 4: Training data with decision boundary (λ = 1)\n', fontsize = 14)
    pyplot.xlabel('Microchip Test1')
    pyplot.ylabel('Microchip Test2')
    pyplot.legend(['y = 1', 'y = 0'], loc='upper right')

plotDecisionBoundary(theta, X, y)

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

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y.flatten()) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')