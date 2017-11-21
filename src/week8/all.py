'''
Created on 20 Nov. 2017

@author: qian.zhu
'''

import numpy
import random
from sklearn.svm import SVC

def generateSample(file, param1, param2):
    X, Y = [], []
    with open(file) as f:
        for line in f.readlines():
            raw = line.strip().split('  ')
            digit = numpy.double(raw[0].strip())
            x1 = numpy.double(raw[1].strip())
            x2 = numpy.double(raw[2].strip())
            if digit == param1:
                X.append([x1, x2])
                Y.append(numpy.double(1.0))
            elif digit == param2 or param2 == 10.0:
                X.append([x1, x2])
                Y.append(numpy.double(-1.0))
    return numpy.array(X), numpy.array(Y)

if __name__ == '__main__':
    print('Problem 2 -------------------------------------------------')
    for i in [0.0, 2.0, 4.0, 6.0, 8.0]:
        clf = SVC(C=0.01, degree=2, gamma=1.0, kernel='poly')
        X, Y = generateSample('./features.train', i, 10.0)
        clf.fit(X, Y)   
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Ein is ' + str(count / len(X)) + ' for ' + str(i) + ' versus all')
        print('Number of Support Vectors is ' + str(sum(clf.n_support_)) +  ' for ' + str(i) + ' versus all')
    print('Problem 3 -------------------------------------------------')
    for i in [1.0, 3.0, 5.0, 7.0, 9.0]:
        clf = SVC(C=0.01, degree=2, gamma=1.0, kernel='poly')
        X, Y = generateSample('./features.train', i, 10.0)
        clf.fit(X, Y)   
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Ein is ' + str(count / len(X)) + ' for ' + str(i) + ' versus all')
        print('Number of Support Vectors is ' + str(sum(clf.n_support_)) +  ' for ' + str(i) + ' versus all')
    print("Problem 5 Q = 2 (1 versus 5) -------------------------------------------------")
    for c in [0.001, 0.01, 0.1, 1]:
        clf = SVC(C=c, degree=2, gamma=1.0, kernel='poly')
        X, Y = generateSample('./features.train', 1.0, 5.0)
        clf.fit(X, Y)
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Ein is ' + str(count / len(X)) + ' for C=' + str(c))
        print('Number of Support Vectors is ' + str(sum(clf.n_support_)) +  ' for C=' + str(c))
        X, Y = generateSample('./features.test', 1.0, 5.0)
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Eout is ' + str(count / len(X)) + ' for C=' + str(c))    
    print("Problem 6 Q = 5 (1 versus 5) -------------------------------------------------")
    for c in [0.001, 0.01, 0.1, 1]:
        clf = SVC(C=c, degree=5, gamma=1.0, kernel='poly')
        X, Y = generateSample('./features.train', 1.0, 5.0)
        clf.fit(X, Y)
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Ein is ' + str(count / len(X)) + ' for C=' + str(c))
        print('Number of Support Vectors is ' + str(sum(clf.n_support_)) +  ' for C=' + str(c))
        X, Y = generateSample('./features.test', 1.0, 5.0)
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Eout is ' + str(count / len(X)) + ' for C=' + str(c))
    print("Problem 9 & 10 (1 versus 5) -------------------------------------------------")
    for c in [0.01, 1.0, 100.0, 10000.0, 1000000.0]:
        clf = SVC(C=c, gamma=1.0, kernel='rbf')
        X, Y = generateSample('./features.train', 1.0, 5.0)
        clf.fit(X, Y)
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1        
        print('Ein is ' + str(count / len(X)) + ' for C=' + str(c))
        X, Y = generateSample('./features.test', 1.0, 5.0)
        count = 0
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                count += 1
        print('Eout is ' + str(count / len(X)) + ' for C=' + str(c))  
    print("Problem 7 & 8 (1 versus 5) -------------------------------------------------")
    record = dict()
    ecv_record = dict()
    for iteration in range(100):
        result = []
        for c in [0.0001, 0.001, 0.01, 0.1, 1]:
            X, Y = generateSample('./features.train', 1.0, 5.0)
            keys = [i for i in range(len(X))]
            random.shuffle(keys)
            division = len(X) / 10.0
            par = [keys[int(round(division * i)): int(round(division * (i + 1)))] for i in range(10)]
            Ecv = 0.0
            for p in par:
                Xcv, Ycv, Xt, Yt, s = [], [], [], [], set(p)
                for index in range(len(X)):
                    if index in s:
                        Xcv.append(X[index])
                        Ycv.append(Y[index])
                    else:
                        Xt.append(X[index])
                        Yt.append(Y[index])                  
                Xcv = numpy.array(Xcv)
                Ycv = numpy.array(Ycv)
                Xt = numpy.array(Xt)
                Yt = numpy.array(Yt)
                clf = SVC(C=c, degree=2, gamma=1.0, kernel='poly')
                clf.fit(Xt, Yt)
                count = 0
                for index in range(len(Xcv)):
                    if numpy.sign(clf.predict([Xcv[index]])) != numpy.sign(Ycv[index]):
                        count += 1
                Ecv += count / len(Xcv)
            Ecv = Ecv / 10
            if c not in ecv_record:
                ecv_record[c] = Ecv
            else:
                ecv_record[c] += Ecv
            result.append((Ecv, c))
        result = sorted(result, key=lambda x: (x[0], x[1]))
        if result[0][1] not in record:
            record[result[0][1]] = 1
        else:
            record[result[0][1]] += 1
    for k in ecv_record.keys():
        ecv_record[k] /= 100
    print(record)
    print(ecv_record)