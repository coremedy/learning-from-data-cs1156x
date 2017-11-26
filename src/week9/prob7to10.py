'''
Created on 26 Nov. 2017

@author: qian.zhu
'''

import numpy
from sklearn import linear_model

def generateSample(file, param1, param2):
    X, Xf, Y = [], [], []
    with open(file) as f:
        for line in f.readlines():
            raw = line.strip().split('  ')
            digit = numpy.double(raw[0].strip())
            x1 = numpy.double(raw[1].strip())
            x2 = numpy.double(raw[2].strip())
            if digit == param1:
                X.append([x1, x2])
                Xf.append([x1, x2, x1 * x2, x1 * x1, x2 * x2])
                Y.append(numpy.double(1.0))
            elif digit == param2 or param2 == 10.0:
                X.append([x1, x2])
                Xf.append([x1, x2, x1 * x2, x1 * x1, x2 * x2])
                Y.append(numpy.double(-1.0))
    return numpy.array(X), numpy.array(Xf), numpy.array(Y)

if __name__ == '__main__':
    print('Problem 7 -------------------------------------------------')
    for i in [5.0, 6.0, 7.0, 8.0, 9.0]:
        X, Xf, Y = generateSample('./features.train', i, 10.0)
        reg = linear_model.Ridge (alpha = 1.0)
        reg.fit(X, Y)
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, X[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print(str(i) + ' versus all (Ein): ' + str(count / len(X)))
    print('Problem 8 -------------------------------------------------')
    for i in [0.0, 1.0, 2.0, 3.0, 4.0]:
        X, Xf, Y = generateSample('./features.train', i, 10.0)
        reg = linear_model.Ridge (alpha = 1.0)
        reg.fit(Xf, Y)
        X, Xf, Y = generateSample('./features.test', i, 10.0)
        count = 0
        for index in range(len(Xf)):
            if numpy.sign(numpy.dot(reg.coef_, Xf[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print(str(i) + ' versus all (Eout): ' + str(count / len(X)))
    print('Problem 9 -------------------------------------------------')
    for i in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        X, Xf, Y = generateSample('./features.train', i, 10.0)
        reg = linear_model.Ridge (alpha = 1.0)
        reg.fit(X, Y)
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, X[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print(str(i) + ' versus all without Z transform (Ein): ' + str(count / len(X))) 
        X, Xf, Y = generateSample('./features.test', i, 10.0)
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, X[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print(str(i) + ' versus all without Z transform (Eout): ' + str(count / len(X)))
        X, Xf, Y = generateSample('./features.train', i, 10.0)
        reg = linear_model.Ridge (alpha = 1.0)
        reg.fit(Xf, Y)        
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, Xf[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print(str(i) + ' versus all with Z transform (Ein): ' + str(count / len(X))) 
        X, Xf, Y = generateSample('./features.test', i, 10.0)
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, Xf[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print(str(i) + ' versus all with Z transform (Eout): ' + str(count / len(X)))
    print('Problem 10 -------------------------------------------------')
    for l in [0.01, 1.0]:
        X, Xf, Y = generateSample('./features.train', 1.0, 5.0)
        reg = linear_model.Ridge (alpha = l)
        reg.fit(Xf, Y)
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, Xf[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print('lambda = ' + str(l) + ', 1 versus 5 (Ein): ' + str(count / len(X)))
        X, Xf, Y = generateSample('./features.test', 1.0, 5.0)
        count = 0
        for index in range(len(X)):
            if numpy.sign(numpy.dot(reg.coef_, Xf[index]) + reg.intercept_) != numpy.sign(Y[index]):
                count += 1
        print('lambda = ' + str(l) + ', 1 versus 5 (Eout): ' + str(count / len(X)))