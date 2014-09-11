#-*- coding: utf-8 -*-
'''
Nonlinear functions

@author: lpeng
'''

import numpy

__all__ = ['softmax', 'sigmoid', 'tanh_norm1_prime', 'sum_along_column']

def softmax(x):
  ''''''
  e_x = numpy.exp(x - x.max(axis=0))
  sm = e_x / e_x.sum(axis=0)
  return sm


def sigmoid(x):
  '''Element-wise sigmoid function'''
  return 1.0 / (1 + numpy.exp(-x))


def tanh_norm1_prime(p_unnormalized):
  '''Borrrowed from Socher's code
  
  Note: p_unnormalized must be column vector, i.e. has the shape (size, 1)
  '''
  x = p_unnormalized
  nrm = numpy.linalg.norm(x)
  y = x-x**3
  return numpy.diag((1-x**2)[:, 0]) / nrm - numpy.dot(y, x.T) / nrm**3


Is = []
def get_I(row_num):
  '''Helper function for computing average
  '''
  while len(Is) < row_num:
    Is.append(numpy.ones(len(Is)+1))
  return Is[row_num-1]


def sum_along_column(matrix):
  '''It is faster than build in sum() 
  '''
  row_num = matrix.shape[0]
  I = get_I(row_num)
  return numpy.dot(I, matrix)