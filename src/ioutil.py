#-*- coding: utf-8 -*-
'''
IO utility

@author: lpeng
'''

import gzip
import codecs
import cPickle as pickle
from os import path, makedirs

__all__ = ['Reader', 'Writer', 'make_parent_dirs']

class DataBuffer(object):
  
  def __init__(self, filename, mode='r', encoding='utf-8'):
    ''' mode : r/w/a '''
    if filename.endswith('.gz'):
      self.source = gzip.open(filename, mode+'b')
      self.next_func = lambda : self.source.next()
      self.read_func = lambda : self.source.read()
      self.readline_func = lambda : self.source.readline()
      self.write_func = lambda x : self.source.write(x)
      self.writelines_func = lambda x : self.source.writelines(x)
    else:
      self.source = codecs.open(filename, mode, encoding)
      self.next_func = lambda : self.source.next().encode(encoding)
      self.read_func = lambda : self.source.read().encode(encoding)
      self.readline_func = lambda : self.source.readline().encode(encoding)
      self.write_func = lambda x : self.source.write(x.decode(encoding))
      self.writelines_func = lambda x : self.source.writelines(
                                        [s.decode(encoding) for s in x])
      
  def __iter__(self):
    return self
  
  def next(self):
    return self.next_func()
  
  def read(self):
    return self.read_func()
  
  def readline(self):
    return self.readline_func()
  
  def write(self, string):
    return self.write_func(string)
  
  def writelines(self, sequence_of_strings):
    return self.writelines_func(sequence_of_strings)
    
  def close(self):
    self.source.close()
  
  def __enter__(self):
    return self
    
  def __exit__(self, exception_type, exception_val, trace):
    try:
        self.close()
    except AttributeError: # obj isn't closable
        print 'Not closable.'
        return True # exception handled successfully

class Reader(DataBuffer):
  def __init__(self, filename, encoding='utf-8'):
    super(Reader, self).__init__(filename, 'r', encoding)
    
class Writer(DataBuffer):
  def __init__(self, filename, encoding='utf-8'):
    super(Writer, self).__init__(filename, 'w', encoding)
    
def make_parent_dirs(filename):
  output_dir = path.dirname(filename)
  if len(output_dir) != 0 and not path.exists(output_dir):
    makedirs(output_dir)
    
def unpickle(filename):
  if filename.endswith('.gz'):
    with gzip.open(filename, 'rb') as theta_reader:
      return pickle.load(theta_reader)
  else:
    with open(filename, 'rb') as theta_reader:
      return pickle.load(theta_reader)
