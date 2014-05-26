'''
Created on May 25, 2014

@author: Sean
'''

import sys
from sys import argv
import numpy as np

sys.stdout = open("HW3-2-output1.txt", 'w')

def add_vectors(a, b):
    '''Add vectors a and b '''
    assert len(a) == len(b)
    return [a[i]+b[i] for i in range(len(a))]

def multiply_scalar_vector(alpha, vec):
    '''Multiply vector vec with scalar alpha '''
    return [alpha*f for f in vec]

def train_classifier(filename):
    train_file = open(filename)
    
    row, col = map(int, train_file.readline().split())
    w_list = []
    y_list = []
    for line in train_file:
        x = list(map(float, line.split()))
        y = x.pop()
        y_list.append(y)
        x.append(float(1))
        w_list.append(x)
    train_file.close()
    w_array = np.asarray(w_list)
    y_array = np.asarray(y_list)
    w_t = w_array.T
    first = np.dot(w_t,w_array)
    second = np.linalg.inv(first)
    third = np.dot(second, w_t)
    last = np.dot(third, y_array)
    print("w, t = ", end='')
    print(last, end=' ')
    print()
    return last
    
def test_classifier(filename, w):
    test_file = open(filename)
    row, col = map(int, test_file.readline().split())
    count = 1
    w_list = w.tolist()
    t = w_list.pop()
    w = np.asarray(w_list)
    for line in test_file:
        x = list(map(float, line.split()))
        x_array = np.asarray(x)
        x_t = x_array.T
        first = np.dot(w, x_t)
        last = first + t
        print(count, end='')
        print('.',end=' ')
        print(x,end=' ')
        print('--',end=' ')
        print(last)
        count = count + 1
    test_file.close()

if __name__ == '__main__':
    if len(argv) == 3:
        temp = train_classifier(argv[1])
        temp2 = test_classifier(argv[2], temp)
    else: 
        print('Please input the training file and the test file')