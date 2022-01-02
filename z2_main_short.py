import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

# Lấy dữ liệu
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

# Biểu diễn các điểm
def plotData(X, y):
    fig = pyplot.figure()

    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'yo', ms=8, mec='k')

# Hàm h
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1/(1 + np.exp(-z))
    return g

# Hàm J
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def costFunction(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
    grad = (1/m) * (np.dot(X.T, (h-y)))
    return J, grad

# Tìm theta
initial_theta = np.zeros(n+1)

options= {'maxiter': 400}

res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)
'''
OtimizeResult
x: kết quả của quá trình tối ưu(ở đây là theta)
'''
theta = res.x

cost = res.fun

print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n')

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

# Vẽ Decision Boundary
utils.plotDecisionBoundary(plotData, theta, X, y)
pyplot.show()

# Dự đoán cho toàn bộ sinh viên, sau khi tìm được theta
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

# Dự đoán cho 1 sinh viên, sau khi tìm được theta
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85, '
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# p là kết quả bằng Decision Boundary, sau khi tìm được theta. y là kết quả thực tế
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')