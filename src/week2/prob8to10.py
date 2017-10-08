'''
Created on 08/10/2017

@author: ken_zhu
'''

import random as ran
import numpy as nu

def genPoint():
    return (ran.uniform(-1, 1), ran.uniform(-1, 1))

def genSamples(count):
    result = dict()
    while count > 0:
        pt = genPoint()
        if (1, pt[0], pt[1]) in result:
            continue
        result[(1, pt[0], pt[1])] = nu.sign(pt[0] * pt[0] + pt[1] * pt[1] - 0.6)
        count -= 1
    randomSet = set(ran.sample(range(0, count), count // 10))
    index = 0
    for key in result.keys():
        if index in randomSet:
            result[key] = -result[key]
        index += 1
    return result

def zTransform(sample):
    result = dict()
    for key in sample.keys():
        result[(1, key[1], key[2], key[1] * key[2], key[1] * key[1], key[2] * key[2])] = sample[key]
    return result

def linearRegression(sample):
    A, B = [], []
    for key in sample.keys():
        A.append(nu.array(key))
        B.append(sample[key])
    A = nu.vstack(A)
    B = nu.array(B)
    return nu.linalg.lstsq(A, B)[0]

if __name__ == '__main__':
    eIn, eOut = 0, 0
    for i in range(1000):
        inSample = genSamples(1000)
        inSamplewithZ = zTransform(inSample)
        outSample = zTransform(genSamples(2000))
        wLinear = linearRegression(inSample)
        for s in inSample:
            eIn = (eIn + 1) if nu.sign(nu.dot(wLinear, s)) != inSample[s] else eIn
        wLinearAfterTransform = linearRegression(inSamplewithZ)
        print(wLinearAfterTransform)
        for s in outSample:
            eOut = (eOut + 1) if nu.sign(nu.dot(wLinearAfterTransform, s)) != outSample[s] else eOut
    print(eIn/(1000 * 1000))
    print(eOut/(2000 * 1000))        