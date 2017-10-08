'''
Created on 08/10/2017

@author: ken_zhu
'''

import random as ran
import numpy as nu

def genPoint():
    return (ran.uniform(-1, 1), ran.uniform(-1, 1))

def genTargetFunction():
    while True:
        pt1, pt2 = genPoint(), genPoint()
        if nu.isclose(pt1[0], pt2[0]) or nu.isclose(pt1[1], pt2[1]):
            continue
        return nu.polyfit(pt1, pt2, 1)

def genSamples(coefficients, count):
    result = dict()
    while count > 0:
        pt = genPoint()
        if nu.isclose(pt[0] * coefficients[0] + coefficients[1], pt[1]):
            continue
        if (1, pt[0], pt[1]) in result:
            continue
        result[(1, pt[0], pt[1])] = 1 if (pt[1] > pt[0] * coefficients[0] + coefficients[1]) else -1
        count -= 1
    return result

def linearRegression(inSample):
    A, B = [], []
    for key in inSample.keys():
        A.append(nu.array(key))
        B.append(inSample[key])
    A = nu.vstack(A)
    B = nu.array(B)
    return nu.linalg.lstsq(A, B)[0]

def calculateFalseEntries(w, samples):
    result = []
    for entry in samples.keys():
        if nu.sign(nu.dot(w, entry)) != samples[entry]:
            result.append(entry)
    return result

def pla(samples, wIn):
    w, iteration = wIn, 0
    falseEntries = calculateFalseEntries(w, samples)
    while len(falseEntries) > 0:
        target = ran.choice(falseEntries)
        w = nu.add(w, samples[target] * nu.array(target)) 
        iteration += 1
        falseEntries = calculateFalseEntries(w, samples)
    return (w, iteration)

if __name__ == '__main__':
    #ein, eout = 0, 0
    #for i in range(1000):
    #    coefficients = genTargetFunction()
    #    inSample = genSamples(coefficients, 100)
    #    outSample = genSamples(coefficients, 1000)
    #    wLinearReg = linearRegression(inSample)
    #    for s in inSample:
    #        ein = (ein + 1) if nu.sign(nu.dot(wLinearReg, s)) != inSample[s] else ein
    #    for s in outSample:
    #        eout = (eout + 1) if nu.sign(nu.dot(wLinearReg, s)) != outSample[s] else eout
    #print(ein/(100 * 1000))
    #print(eout/(1000 * 1000))
    
    totalIteration = 0
    for i in range(1000):
        coefficients = genTargetFunction()
        inSample = genSamples(coefficients, 10)
        wLinearReg = linearRegression(inSample)
        totalIteration += pla(inSample, wLinearReg)[1]
    print(totalIteration/1000)