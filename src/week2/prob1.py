'''
Created on 08/10/2017

@author: ken_zhu
'''

import random

if __name__ == '__main__':
    result = 0
    for i in range(0, 100000):
        result += min([(sum([random.sample(range(0,2), 1)[0] for flip in range(10)]) / 10) for coin in range(0, 1000)])
        print(result / (i + 1))