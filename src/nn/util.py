#-*- coding: utf-8 -*-
'''
Neural network utilities

@author: lpeng
'''
from numpy import sqrt
from numpy.random import rand

__all__ = ['init_W', 'init_We']

def init_W(row, col, return_row_vector=True):
  '''Initialize weight matrix.
  
  Args:
    row: number of rows
    col: number of columns
    return_row_vector: whether return a row vector or not
    
  Returns:
    An numpy.array of size row*col 
  '''
  r = sqrt(6) / sqrt((row+1) + (col+1)) # 1 : for bias
  if return_row_vector:
    return rand(row*col) * 2 * r - r
  else:
    return rand(row, col) * 2 * r - r

  
def init_We(emb_size, vcb_size, r=0.05, return_row_vector=True):
  '''Initialize word embedding matrix.
  
  Args:
    emb_size: dimension of word embedding vectors
    vcb_size: number of words
    r: the elements of the matrix will be sampled uniformly from [-r, r]
    return_row_vector: whether return a row vector or not
    
  Returns:
    An numpy.array of size emb_size*vcb_size
  '''
  if return_row_vector:
    return rand(emb_size*vcb_size) * 2 * r - r
  else:
    return rand(emb_size, vcb_size) * 2 * r - r
