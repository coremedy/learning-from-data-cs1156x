'''
Created on 05/11/2017

@author: ken_zhu
'''

import numpy as nu

def genSampleData(file):
    A, B = [], []
    with open(file, 'r') as f:
        for line in f:
            x = nu.array(line.strip().split('  ')).astype(nu.float64)
            A.append(nu.array([nu.float64(1), x[0], x[1], x[0] * x[0], x[1] * x[1], x[0] * x[1], nu.abs(x[0] - x[1]), nu.abs(x[0] + x[1])]))
            B.append(x[2])
    return (nu.vstack(A), nu.array(B))

def linearRegression(A, B):
    return nu.linalg.lstsq(A, B)[0]

def weightDecay(A, B, LAMBDA):
    A = nu.matrix(A)
    At = A.copy().transpose()
    B = nu.matrix(B)
    B = B.transpose()
    I = LAMBDA * nu.identity(8)
    return nu.array(nu.dot(nu.dot(nu.linalg.inv(nu.dot(At, A) + I), At), B).transpose())[0]

if __name__ == '__main__':
    (Ain, Bin) = genSampleData('in.dta')
    (Aout, Bout) = genSampleData('out.dta')
    wInitial = linearRegression(Ain, Bin)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wInitial)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (before weight decay): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wInitial)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (before weight decay): ' + str(count / len(Bout))) 
    wDecay2 = weightDecay(Ain, Bin, 1000)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay2)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 1000): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay2)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 1000): ' + str(count / len(Bout)))
    wDecay3 = weightDecay(Ain, Bin, 100)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay3)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 100): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay3)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 100): ' + str(count / len(Bout)))
    wDecay4 = weightDecay(Ain, Bin, 10)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay4)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 10): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay4)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 10): ' + str(count / len(Bout)))
    wDecay5 = weightDecay(Ain, Bin, 1)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay5)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 1): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay5)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 1): ' + str(count / len(Bout)))
    wDecay6 = weightDecay(Ain, Bin, 0.1)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay6)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 0.1): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay6)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 0.1): ' + str(count / len(Bout)))
    wDecay7 = weightDecay(Ain, Bin, 0.01)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay7)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 0.01): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay7)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 0.01): ' + str(count / len(Bout)))
    wDecay1 = weightDecay(Ain, Bin, 0.001)
    index = 0
    count = 0
    for row in Ain:
        if nu.sign(nu.dot(row, wDecay1)) != nu.sign(Bin[index]):
            count += 1
        index += 1
    print('Ein  (weight decay 0.001): ' + str(count / len(Bin)))
    index = 0
    count = 0    
    for row in Aout:
        if nu.sign(nu.dot(row, wDecay1)) != nu.sign(Bout[index]):
            count += 1
        index += 1
    print('Eout (weight decay 0.001): ' + str(count / len(Bout)))