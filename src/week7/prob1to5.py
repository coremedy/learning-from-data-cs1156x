'''
Created on 13/11/2017

@author: ken_zhu

'''

import numpy

def genSampleData(file, k):
    A, B = [], []
    with open(file, 'r') as f:
        for line in f:
            x = numpy.array(line.strip().split('  ')).astype(numpy.float64)
            
            A.append(numpy.array([numpy.float64(1), x[0], x[1], x[0] * x[0], x[1] * x[1], x[0] * x[1], numpy.abs(x[0] - x[1]), numpy.abs(x[0] + x[1])][:k+1]))
            B.append(x[2])
    return (numpy.vstack(A), numpy.array(B))

def linearRegression(A, B):
    return numpy.linalg.lstsq(A, B)[0]

if __name__ == '__main__':
    print("problem 1 - 2")
    for k in range(3, 8):
        (Ain, Bin) = genSampleData('in.dta', k)
        (Aout, Bout) = genSampleData('out.dta', k)
        wInitial = linearRegression(Ain[:25], Bin[:25])
        print("k = " + str(k))
        count = 0
        for index in range(25, len(Ain)):
            if numpy.sign(numpy.dot(wInitial, Ain[index])) != numpy.sign(Bin[index]):
                count += 1
        print("Eval = " + str(count / 10))
        count = 0
        for index in range(len(Aout)):
            if numpy.sign(numpy.dot(wInitial, Aout[index])) != numpy.sign(Bout[index]):
                count += 1
        print("Etest = " + str(count / len(Aout)))    
    print("problem 3 - 4")    
    for k in range(3, 8):
        (Ain, Bin) = genSampleData('in.dta', k)
        (Aout, Bout) = genSampleData('out.dta', k)
        wInitial = linearRegression(Ain[25:], Bin[25:])
        print("k = " + str(k))
        count = 0
        for index in range(0, 24):
            if numpy.sign(numpy.dot(wInitial, Ain[index])) != numpy.sign(Bin[index]):
                count += 1
        print("Eval = " + str(count / 10))
        count = 0
        for index in range(len(Aout)):
            if numpy.sign(numpy.dot(wInitial, Aout[index])) != numpy.sign(Bout[index]):
                count += 1
        print("Etest = " + str(count / len(Aout)))    