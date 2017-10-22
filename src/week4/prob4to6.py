'''
Created on 22/10/2017

@author: ken_zhu
'''

import random as ran
import numpy as nu

def genPoints():
    s = set()
    s.add(ran.uniform(-1, 1))
    while True:
        x = ran.uniform(-1, 1)
        if x not in s:
            s.add(x)
            break
    A = nu.array(list(s))
    B = nu.array([nu.sin(value * nu.pi) for value in A])
    return [A, B]

def linearRegressionThruOrigin(A, B):
    return nu.linalg.lstsq(A[:, nu.newaxis], B)[0]



if __name__ == '__main__':
    g, pts = [], []
    for i in range(10000):
        pts.append(genPoints())
        g.append(linearRegressionThruOrigin(pts[-1][0], pts[-1][1])[0])
    gBar = (sum(g) / 10000)
    print(gBar)
    rawBias = sum(nu.square(a - gBar) for a in g) / 10000
    print(rawBias/2) # rawBias' * x^2 / 4 -> 1 and -1
    def rawVar(x):
        return nu.square(gBar * x - nu.sin(nu.pi * x)) * 0.5
    print(nu.sum(rawVar(nu.linspace(-1, 1, 10000)))* 2 / 10000)