'''
Created on 29/09/2017

@author: ken_zhu

Visit this link (http://www.lfd.uci.edu/~gohlke/pythonlibs/) if you need Python packages on Windows.
pip install wheel from Script folder of Python installation
pip install *.whl

Steps:

1) Area [-1, +1] * [-1, +1]
2) Choose two points and form one line (which is f)
3) The area is divided with -1 and +1
4) Choose the required number of samples (10 and 100)
5) Start the first iteration with weight vector w (all zeros) and record the number of iterations until w converges
6) Calculate P[f(x) != g(x)] with monte carlo method
7) Repeat the following steps 1000 times
'''

import random as ran
import numpy as nu
import matplotlib.pyplot as plt

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

def calculateFalseEntries(w, samples):
    result = []
    for entry in samples.keys():
        if nu.sign(nu.dot(w, entry)) != samples[entry]:
            result.append(entry)
    return result

def pla(samples):
    w, iteration = [0, 0, 0], 0
    falseEntries = calculateFalseEntries(w, samples)
    while len(falseEntries) > 0:
        target = ran.choice(falseEntries)
        w = nu.add(w, samples[target] * nu.array(target)) 
        iteration += 1
        falseEntries = calculateFalseEntries(w, samples)
    return (w, iteration)

if __name__ == '__main__':
    #count = []
    #for i in range(1000):
    #    coefficients = genTargetFunction()
    #    learningSamples = genSamples(coefficients, 10)        
    #    count.append(pla(learningSamples)[1])
    #plt.hist(count, 10, range=[1, 50], facecolor='blue', align='mid')
    #plt.xlabel('Mean value: ' + str(sum(count) / 1000))
    #plt.ylabel('Occurrence')
    #plt.show()    
    
    #count = []
    #for i in range(1000):
    #    coefficients = genTargetFunction()
    #    learningSamples = genSamples(coefficients, 100)        
    #    count.append(pla(learningSamples)[1])
    #plt.hist(count, 20, range=[1, 200], facecolor='blue', align='mid')
    #plt.xlabel('Mean value: ' + str(sum(count) / 1000))
    #plt.ylabel('Occurrence')
    #plt.show()
    
    #missedArray = []
    #for i in range(1000):
    #    coefficients = genTargetFunction()
    #    learningSamples = genSamples(coefficients, 100)
    #    mcSamples = genSamples(coefficients, 1000)
    #    w, count = pla(learningSamples)
    #    missed = 0
    #    for k in mcSamples.keys():
    #        if nu.sign(nu.dot(w, k)) != mcSamples[k]:
    #            missed += 1
    #    missedArray.append(missed/len(mcSamples))
    #print(sum(missedArray)/1000)    
    
    missedArray = []
    for i in range(1000):
        coefficients = genTargetFunction()
        learningSamples = genSamples(coefficients, 100)
        mcSamples = genSamples(coefficients, 1000)
        w, count = pla(learningSamples)
        missed = 0
        for k in mcSamples.keys():
            if nu.sign(nu.dot(w, k)) != mcSamples[k]:
                missed += 1
        missedArray.append(missed/len(mcSamples))
    print(sum(missedArray)/1000)