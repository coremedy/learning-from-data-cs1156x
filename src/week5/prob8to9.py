'''
Created on 29/10/2017

@author: ken_zhu
'''

import random as ran
import numpy as nu
import itertools

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

if __name__ == '__main__':
    runs = 100
    overallCrossEntropyError = 0
    overallEpoch = 0
    while runs > 0:
        sample = genSamples(genTargetFunction(), 1100)
        i = iter(sample.items())
        inSample = dict(itertools.islice(i, 100))
        outSample = dict(i)
        w = nu.array([nu.float64(0.0), nu.float64(0.0), nu.float64(0.0)])
        epoch = 0
        while True:
            wPrev = nu.array(list(w))
            epoch += 1
            candidate = list(inSample.keys())
            ran.shuffle(candidate)
            for cand in candidate:
                denominator = nu.float64(1.0) + nu.exp(nu.dot(nu.array(cand), w) * inSample[cand])                
                wDelta = nu.array(cand) * nu.float64(0.01) * nu.float64(inSample[cand]) / denominator             
                w = nu.add(w, wDelta)
            # Attention - Frobenius norm
            if nu.sqrt(nu.sum((wPrev - w)**2)) < 0.01:
                break
        crossEntropyError = 0;
        for key in outSample:
            crossEntropyError += nu.log(nu.float64(1.0) + nu.exp(nu.float64(-1) * outSample[key] * nu.dot(w, key)))
        overallCrossEntropyError += crossEntropyError / 1000
        overallEpoch += epoch
        runs -= 1
    print(overallCrossEntropyError / 100)
    print(overallEpoch / 100)
    