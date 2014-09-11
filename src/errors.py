#-*- coding: utf-8 -*-
'''
Exception classes

@author: lpeng
'''

class UnexpectedStateError(Exception):
  
  def __init__(self, msg):
    self.msg = msg
    
  def __str__(self):
    return self.msg
  
class UnsupportedOperationError(Exception):
  
  def __init(self, msg):
    self.msg = msg
    
  def __str__(self):
    return self.msg
  
class GridentCheckingFailedError(Exception):
  
  def __init(self, msg):
    self.msg = msg
    
  def __str__(self):
    return self.msg