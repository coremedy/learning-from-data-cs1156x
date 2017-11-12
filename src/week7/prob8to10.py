'''
Created on 12/11/2017

@author: ken_zhu

http://goelhardik.github.io/2016/11/28/svm-cvxopt/
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
'''

import numpy
from cvxopt import matrix
from cvxopt import solvers
import random
import itertools

def genPoint():
    return (random.uniform(-1, 1), random.uniform(-1, 1))

def genTargetFunction():
    while True:
        pt1, pt2 = genPoint(), genPoint()
        if numpy.isclose(pt1[0], pt2[0]) or numpy.isclose(pt1[1], pt2[1]):
            continue
        return numpy.polyfit(pt1, pt2, 1)

def genSamples(coefficients, count):
    result = dict()
    while count > 0:
        pt = genPoint()
        if numpy.isclose(pt[0] * coefficients[0] + coefficients[1], pt[1]):
            continue
        if (1, pt[0], pt[1]) in result:
            continue
        result[(1, pt[0], pt[1])] = 1 if (pt[1] > pt[0] * coefficients[0] + coefficients[1]) else -1
        count -= 1
    return result

def svmQP(inSample):
    NUM = len(inSample)
    x = []
    y = []
    for key in inSample:
        x.append([numpy.float64(key[1]), numpy.float64(key[2])])
        y.append(numpy.float64(inSample[key]))
    x = numpy.array(x)
    y = numpy.array(y)
    p = []
    for i in range(len(inSample)):
        row = []
        for j in range(len(inSample)):
            row.append(numpy.float64(y[i] * y[j] * numpy.dot(x[i], x[j])))
        p.append(row)
    P = matrix(p)
    q = matrix(-numpy.ones((NUM, 1)))
    G = matrix(-numpy.eye(NUM))
    h = matrix(numpy.zeros(NUM))    
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix(numpy.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = numpy.array(sol['x'])
    w = numpy.sum(alphas * y[:, None] * x, axis = 0)
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - numpy.dot(x[cond], w)
    bias = b[0]
    return [bias, w[0], w[1]]

def calculateFalseEntries(w, samples):
    result = []
    for entry in samples.keys():
        if numpy.sign(numpy.dot(w, entry)) != samples[entry]:
            result.append(entry)
    return result

def pla(samples):
    w, iteration = [0, 0, 0], 0
    falseEntries = calculateFalseEntries(w, samples)
    while len(falseEntries) > 0:
        target = random.choice(falseEntries)
        w = numpy.add(w, samples[target] * numpy.array(target)) 
        iteration += 1
        falseEntries = calculateFalseEntries(w, samples)
    return (w, iteration)

if __name__ == '__main__':
    better = 0
    iteration = 0
    while iteration <= 1000:
        target = genTargetFunction()
        sample = genSamples(target, 1010)
        i = iter(sample.items())
        inSample = dict(itertools.islice(i, 10))
        testSample = dict(i)
        if len(set(inSample.values())) < 2:
            continue
        wPLA = pla(inSample)[0]
        count = 0
        for key in testSample:
            if numpy.sign(numpy.dot(key, wPLA)) != numpy.sign(testSample[key]):
                count += 1
        rPLA = count / len(testSample)
        wSVM = svmQP(inSample)
        count = 0
        for key in testSample:
            if numpy.sign(numpy.dot(key, wSVM)) != numpy.sign(testSample[key]):
                count += 1
        rSVM = count / len(testSample)
        if rPLA > rSVM:
            better += 1
        iteration += 1
        print(iteration)
        print(better/iteration)
