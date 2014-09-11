#-*- coding: utf-8 -*-
'''
Test suite for nn.functions

@author: lpeng
'''
import unittest
import numpy as np
from functions import sum_along_column, tanh_norm1_prime, softmax

class FunctionsTestCase(unittest.TestCase):
  
  def test_softmax(self):
    x = np.array([[1, 2, 3]]).T
    expected = np.array([[0.0900, 0.2447, 0.6652]]).T
    y = softmax(x)
    diff_abs = abs(y - expected)
    self.assertFalse((diff_abs > 0.0001).any())
  
  def test_sum_along_column(self):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    x_sum = sum_along_column(x) 
    x_sum_expected = np.array([5, 7, 9])

    self.assertFalse((x_sum != x_sum_expected).any())
    
  def test_tanh_norm1_prime(self):
    x = np.array([[0.5345, 1.0690, 1.6036]]).T
    prime = tanh_norm1_prime(x)
    prime_expected = np.array([[0.3316, -0.0510, -0.0765],
                               [0.0102, -0.0510,  0.0306],
                               [0.1684,  0.3367, -0.2806]])
    diff_abs = abs(prime - prime_expected)
    self.assertFalse((diff_abs > 0.0001).any())