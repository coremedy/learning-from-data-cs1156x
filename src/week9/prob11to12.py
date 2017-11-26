'''
Created on 26 Nov. 2017

@author: qian.zhu
'''

import numpy
from sklearn.svm import SVC

if __name__ == '__main__':
    X = numpy.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0], [-2.0, 0.0]])
    Y = numpy.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    print('Problem 11 -------------------------------------------------')
    Xt = []
    for row in X:
        Xt.append([row[1] * row[1] - 2.0 * row[0] - 1, row[0] * row[0] - 2 * row[1] + 1])
    Xt = numpy.array(Xt)
    clf = SVC(kernel='linear')
    clf.fit(Xt, Y)
    print(clf._get_coef())
    print('Problem 12 -------------------------------------------------')
    clf = SVC(C=1000, degree=2, gamma=1.0, kernel='poly')
    clf.fit(X, Y)
    print(sum(clf.n_support_))