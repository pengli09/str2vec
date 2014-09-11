#-*- coding: utf-8 -*-
'''
Created on May 11, 2014

@author: lpeng
'''
import numpy as np

from ioutil import Reader

class WordVectors(object):
  
  def __init__(self, embsize):
    self._vectors = np.array([[0]]) # add a place holder for OOV
    self._word2id = {'OOV':0}
    self._embsize = embsize
  
  def __len__(self):
    return len(self._word2id)
  
  def embsize(self):
    return self._embsize
  
  def __getitem__(self, index_or_index_array):
    return self._vectors[:, index_or_index_array]
  
  def get_word_index(self, word):
    return self._word2id.get(word, 0) 

  @classmethod
  def load_vectors(cls, filename):
    '''
    Load word vectors from a file
    
    Args:
      filename: the name of the file that contains the word vectors
        Comment lines are started with #
        If the first line except comments contains only two integers, it's
        assumed that the first is the vocabulary size and the second is the 
        word embedding size (the same as word2vec).
        
    Return:
      a class of word vectors
    '''
    at_beginning = True
    with Reader(filename) as f:
      idx = 1 # 0 for OOV
    
      vectors = [[0]] # placeholder for OOV
      word2id = {'OOV':0}
      
      for line in f:
        if line.startswith('#'):
          continue
        
        if at_beginning:
          at_beginning = False
          parts = line.strip().split()
          if len(parts) == 2:
            embsize = int(parts[1])
            oov = np.zeros(embsize)
          else:
            word = parts[0]
            vec = np.array([float(v) for v in parts[1:]])
            embsize = len(vec)
            oov = np.zeros(embsize)
            oov += vec
            vectors.append(vec)
            word2id[word] = idx
            idx += 1
        else:
          parts = line.strip().split(' ');
          word = parts[0]
          vec = np.array([float(v) for v in parts[1:]])
          assert(vec.size == embsize)
          oov += vec
          vectors.append(vec)
          word2id[word] = idx
          idx += 1
        
      oov = oov / (len(vectors)-1)
      vectors[0] = oov
      
      word_vectors = WordVectors(embsize)
      word_vectors._vectors = np.array(vectors).T
      word_vectors._word2id = word2id
      
      return word_vectors


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('word_vector_file')
  options = parser.parse_args()
  
  word_vector_file = options.word_vector_file
  word_vectors = WordVectors.load_vectors(word_vector_file)
  
  print word_vectors[[1,2]]
  print len(word_vectors)
  print word_vectors._word2id
  print word_vectors._vectors