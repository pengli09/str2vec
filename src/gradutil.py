#-*- coding: utf-8 -*-
'''
Grident checking tool

@author: lpeng
'''
from sys import stderr
from math import sqrt
from numpy import zeros
from numpy.linalg import norm

def check_grad(func, x0, func_prime=None, args=(), level=1e-5, verbose=0):
  ''' if func_prime is None, func must return grad. 
  '''  
  if func_prime == None: 
    _, grad_actual = func(x0, *args)
  else:
    grad_actual = func_prime(x0, *args)
    
  grad_numerical = numerical_grad(func, x0, grad_actual, 
                                  with_grad=func_prime is None,
                                  args=args, level=level,
                                  verbose=verbose)
  
  diff = abs(grad_actual-grad_numerical)
  
  bad = (diff > level).any()
  if bad:
    print >> stderr, 'Grads = '
    print >> stderr, '%22s\t%20s\t%20s' % ('Index', 'Actual', 'Numerical')
    for index, actual, numerical in zip(xrange(0,len(grad_actual)), 
                                        grad_actual,
                                        grad_numerical):
      print >> stderr, '  %20d\t%20.8f\t%20.8f' % (index, actual, numerical)  

  return not bad

def numerical_grad(func, x0, grad_actual, with_grad=True, args=(), level=1e-5,
                   verbose=0):
  sz = x0.size
  #_mu = 2*sqrt(1e-12)*(1+norm(x0))/sz
  _mu = max(1e-5, 2*sqrt(1e-12)*(1+norm(x0)))
  if verbose >= 1:
    print '      _mu: %s\n' % _mu
  f0_delta1 = zeros(x0.shape)
  f0_delta2 = zeros(x0.shape)
  for i in xrange(0, x0.size):
    if i % 100 == 0:
      print >> stderr, "Checking x0[%d]" % i
    mu_i = zeros(sz)
    mu_i[i] = _mu
    if with_grad:
      f0_delta1[i], _ = func(x0+mu_i, *args)
      f0_delta2[i], _ = func(x0-mu_i, *args)
    else:
      f0_delta1[i] = func(x0+mu_i, *args)
      f0_delta2[i] = func(x0-mu_i, *args)
    if verbose >= 3:
      print >> stderr, ' f0_delta1: %s' % f0_delta1
      print >> stderr, ' f0_delta2: %s' % f0_delta2
      print >> stderr, '     mu_i: %s' % mu_i
      print >> stderr, '  x0+mu_i: %s' % (x0+mu_i)
      print >> stderr, ''
    
    gradi = (f0_delta1[i]-f0_delta2[i])/(2*_mu)
    if abs(gradi-grad_actual[i]) > level:
      print >> stderr, (' diff at %d -> actual: %20.8f, numerical: %20.8f' 
                        % (i, grad_actual[i], gradi)) 
    if verbose >= 2:
      print (' grad at %d -> actual: %20.8f, numerical: %20.8f' 
             % (i, grad_actual[i], gradi))
    
  delta = f0_delta1 - f0_delta2
  grad = delta / (2*_mu)
  if verbose >= 3:
    print '    delta: %s' % delta
  return grad