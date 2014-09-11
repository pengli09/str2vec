#-*- coding: utf-8 -*-
'''
L-BFGS wrapper

@author: lpeng
'''

import sys

import scipy.optimize as sopt

import gradutil
from errors import GridentCheckingFailedError

maxiter=10
iprint=1 

def optimize(func, x0, maxiter, verbose=1, check_grad=False, args=(),
             callback=None):
  options = {'maxiter':maxiter}
  if verbose:
    # print info. to stdout every iprint iterations
    options['iprint'] = 1
  
  if check_grad:
    print >> sys.stderr, "Check gradients"
    _pass = gradutil.check_grad(func, x0, args=args, verbose=verbose)
    if _pass:
      print >> sys.stderr, 'Pass'
    else:
      raise GridentCheckingFailedError('Gradient checking failed')
  
  ret = sopt.minimize(func, x0, args, method='L-BFGS-B',
                      jac=True, options=options, callback=callback)
  if verbose >= 1:
    print '%10s: %d' % ('status', ret.status)
    print '%10s: %s' % ('success', ret.success)
    print '%10s: %d' % ('funcalls', ret.nfev)  
    print '%10s: %s' % ('fun', ret.fun)
    print '%10s: %s' % ('message', ret.message)
    print '%10s: %d' % ('nit', ret.nit)
    
  return ret.x