#-*- coding: utf-8 -*-
'''
MPI signals

@author: lpeng
'''

class TerminatorSignal(object):
  '''Terminate the programm gracefully'''
  pass


class WorkingSignal(object):
  '''Go to work'''
  pass


class ForceQuitSignal(object):
  '''Force quit'''
  pass

