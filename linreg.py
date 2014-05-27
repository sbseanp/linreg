'''
Created on May 25, 2014

@author: Sean
'''

from sys import argv
import numpy as np

#sys.stdout = open("HW3-2-output2.txt", 'w')

'''return w and t'''
def train_classifier(filename):
    train_file = open(filename)
    '''get MxN'''
    row, col = map(int, train_file.readline().split())
    '''go line by line and add w's and y's to their respective lists'''
    w_list = []
    y_list = []
    for line in train_file:
        x = list(map(float, line.split()))
        y = x.pop()
        y_list.append(y)
        '''make homogeneous'''
        x.append(float(1))
        w_list.append(x)
    train_file.close()
    '''make list into numpy array'''
    w_array = np.asarray(w_list)
    y_array = np.asarray(y_list)
    '''get transpose of w'''
    w_t = w_array.T
    '''use formula (K^T*K)^-1*K^T*y)'''
    first = np.dot(w_t,w_array)
    second = np.linalg.inv(first)
    third = np.dot(second, w_t)
    last = np.dot(third, y_array)
    '''print out regression values'''
    print("w, t = ", end='')
    print(last, end=' ')
    print()
    return last
    
def test_classifier(filename, w):
    test_file = open(filename)
    '''get MxN'''
    row, col = map(int, test_file.readline().split())
    '''make w back into list to pop off the t value then make it an array again'''
    w_list = w.tolist()
    t = w_list.pop()
    w = np.asarray(w_list)
    count = 1
    '''go line by line and use formula y = w^T*x + t'''
    for line in test_file:
        x = list(map(float, line.split()))
        x_array = np.asarray(x)
        x_t = x_array.T
        first = np.dot(w, x_t)
        last = first + t
        '''print output with sample formatting'''
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
        test_classifier(argv[2], temp)
    else: 
        print('Please input the training file and the test file')