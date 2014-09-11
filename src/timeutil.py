#-*- coding: utf-8 -*-
'''
Created on Apr 18, 2014

@author: lpeng
'''
import time

class Timer(object):
  
  def tic(self):
    self.start = time.time()
    
  def toc(self):
    return time.time() - self.start