'''
Created on 29/10/2017

@author: ken_zhu
'''

import numpy as nu

def errorSurface(u, v):
    return nu.square(u * nu.exp(v) - nu.float64(2.0) * v * nu.exp(-u))

def uPartialDerivate(u, v):
    return nu.float64(2.0) * (u * nu.exp(v) - nu.float64(2.0) * v * nu.exp(-u)) * (nu.exp(v) + nu.float64(2.0) * v * nu.exp(-u))

def vPartialDerivate(u, v):
    return nu.float64(2.0) * (u * nu.exp(v) - nu.float64(2.0) * v * nu.exp(-u)) * (u * nu.exp(v) - nu.float64(2.0) * nu.exp(-u))

if __name__ == '__main__':
    u, v, count = nu.float64(1.0), nu.float64(1.0), 0
    while not nu.isclose(errorSurface(u, v), nu.float64(nu.power(10, -14))):
        uNew = u - nu.float64(0.1) * uPartialDerivate(u, v)
        vNew = v - nu.float64(0.1) * vPartialDerivate(u, v)
        u, v = uNew, vNew
        count += 1
    print(count)
    print(u, v)
    print("coordinate descent")
    u, v, count = nu.float64(1.0), nu.float64(1.0), 15
    while count > 0:
        uNew = u - nu.float64(0.1) * uPartialDerivate(u, v)
        vNew = v - nu.float64(0.1) * vPartialDerivate(uNew, v)
        u, v = uNew, vNew
        count -= 1
    print(errorSurface(u, v))