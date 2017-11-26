'''
Created on 26 Nov. 2017

@author: qian.zhu
'''

import random
import numpy
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def genPoints(train, test):
    count, s = 0, set()
    X, Y, Xt, Yt = [], [], [], []
    while count < train + test:
        x1 = numpy.float64(random.uniform(0, 1))
        x2 = numpy.float64(random.uniform(0, 1))
        if (x1, x2) in s:
            continue
        else:
            s.add((x1, x2))
        y = numpy.float64(numpy.sign(x2 - x1 + 0.25 * numpy.sin(numpy.pi + x1)))
        if count < train:
            X.append([x1, x2])
            Y.append(y)
        else:
            Xt.append([x1, x2])
            Yt.append(y)
        count += 1
    return (numpy.array(X), numpy.array(Y), numpy.array(Xt), numpy.array(Yt))

if __name__ == '__main__':
    print('Problem 13 -------------------------------------------------')
    fail = 0
    for i in range(1000):
        X, Y, Xt, Yt = genPoints(100, 100)
        clf = SVC(C=10000, gamma=1.5, kernel='rbf')
        clf.fit(X, Y)   
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                fail += 1
                break
    print(fail / 1000) 
    print('Problem 14 -------------------------------------------------')
    win = 0
    count = 0
    for i in range(1000):
        skip = False
        X, Y, Xt, Yt = genPoints(100, 1000)
        clf = SVC(C=10000, gamma=1.5, kernel='rbf')
        clf.fit(X, Y)   
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                skip = True
                break
        if skip:
            continue
        EoutSVM = 0
        for index in range(len(Xt)):
            if numpy.sign(clf.predict([Xt[index]])) != numpy.sign(Yt[index]):
                EoutSVM += 1
        EoutSVM = EoutSVM / 1000
        kmeans = KMeans(n_clusters=9, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 9:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EoutRBF = 0
        for index in range(len(Xt)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(Xt[index], c), numpy.subtract(Xt[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Yt[index]):
                EoutRBF += 1
        EoutRBF = EoutRBF / 1000
        if EoutSVM < EoutRBF:
            win += 1
        count += 1
    print(win / count)
    print('Problem 15 -------------------------------------------------')
    win = 0
    count = 0
    for i in range(1000):
        skip = False
        X, Y, Xt, Yt = genPoints(100, 1000)
        clf = SVC(C=10000, gamma=1.5, kernel='rbf')
        clf.fit(X, Y)   
        for index in range(len(X)):
            if numpy.sign(clf.predict([X[index]])) != numpy.sign(Y[index]):
                skip = True
                break
        if skip:
            continue
        EoutSVM = 0
        for index in range(len(Xt)):
            if numpy.sign(clf.predict([Xt[index]])) != numpy.sign(Yt[index]):
                EoutSVM += 1
        EoutSVM = EoutSVM / 1000
        kmeans = KMeans(n_clusters=12, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 12:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EoutRBF = 0
        for index in range(len(Xt)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(Xt[index], c), numpy.subtract(Xt[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Yt[index]):
                EoutRBF += 1
        EoutRBF = EoutRBF / 1000
        if EoutSVM < EoutRBF:
            win += 1
        count += 1
    print(win / count)
    print('Problem 16 -------------------------------------------------')
    for i in range(10):
        X, Y, Xt, Yt = genPoints(100, 1000)
        kmeans = KMeans(n_clusters=9, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 9:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EinRBF = 0
        for index in range(len(X)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(X[index], c), numpy.subtract(X[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Y[index]):
                EinRBF += 1
        EinRBF = EinRBF / 100        
        print('Cluster = 9 Ein = ' + str(EinRBF))
        EoutRBF = 0
        for index in range(len(Xt)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(Xt[index], c), numpy.subtract(Xt[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Yt[index]):
                EoutRBF += 1
        EoutRBF = EoutRBF / 1000
        print('Cluster = 9 Eout = ' + str(EoutRBF))
        kmeans = KMeans(n_clusters=12, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 12:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EinRBF = 0
        for index in range(len(X)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(X[index], c), numpy.subtract(X[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Y[index]):
                EinRBF += 1
        EinRBF = EinRBF / 100        
        print('Cluster = 12 Ein = ' + str(EinRBF))
        EoutRBF = 0
        for index in range(len(Xt)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(Xt[index], c), numpy.subtract(Xt[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Yt[index]):
                EoutRBF += 1
        EoutRBF = EoutRBF / 1000        
        print('Cluster = 12 Eout = ' + str(EoutRBF)) 
    print('Problem 17 -------------------------------------------------')
    for i in range(10):
        X, Y, Xt, Yt = genPoints(100, 1000)
        kmeans = KMeans(n_clusters=9, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 9:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EinRBF = 0
        for index in range(len(X)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(X[index], c), numpy.subtract(X[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Y[index]):
                EinRBF += 1
        EinRBF = EinRBF / 100        
        print('Cluster = 9 gamma = 1.5 Ein = ' + str(EinRBF))
        EoutRBF = 0
        for index in range(len(Xt)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(Xt[index], c), numpy.subtract(Xt[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Yt[index]):
                EoutRBF += 1
        EoutRBF = EoutRBF / 1000
        print('Cluster = 9 gamma = 1.5 Eout = ' + str(EoutRBF))
        kmeans = KMeans(n_clusters=9, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 9:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 2 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EinRBF = 0
        for index in range(len(X)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 2 * numpy.dot(numpy.subtract(X[index], c), numpy.subtract(X[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Y[index]):
                EinRBF += 1
        EinRBF = EinRBF / 100        
        print('Cluster = 9 gamma = 2 Ein = ' + str(EinRBF))
        EoutRBF = 0
        for index in range(len(Xt)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 2 * numpy.dot(numpy.subtract(Xt[index], c), numpy.subtract(Xt[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Yt[index]):
                EoutRBF += 1
        EoutRBF = EoutRBF / 1000        
        print('Cluster = 9 gamma = 2 Eout = ' + str(EoutRBF))
    print('Problem 18 -------------------------------------------------') 
    count  = 0
    for i in range(1000):
        X, Y, Xt, Yt = genPoints(100, 1000)
        kmeans = KMeans(n_clusters=9, init='random').fit(X)
        if len(kmeans.cluster_centers_) != 9:
            continue
        M = []
        for x in X:
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(x, c), numpy.subtract(x, c))))
            row.append(1.0)
            M.append(row)
        w = numpy.dot(numpy.linalg.pinv(M), Y)
        EinRBF = 0
        for index in range(len(X)):
            row = []
            for c in kmeans.cluster_centers_:
                row.append(numpy.exp(-1.0 * 1.5 * numpy.dot(numpy.subtract(X[index], c), numpy.subtract(X[index], c))))
            row.append(1.0)
            if numpy.sign(numpy.dot(w, row)) != numpy.sign(Y[index]):
                EinRBF += 1
        EinRBF = EinRBF / 100
        if EinRBF == 0.0:
            count += 1
    print(count / 1000)