import sys
import cPickle as pickle
import argparse
from ioutil import Reader, Writer
import gzip

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('theta')
  parser.add_argument('output')
  options = parser.parse_args()

  theta_file = options.theta
  output_file = options.output

  if theta_file.endswith('.gz'):
    with gzip.open(theta_file, 'rb') as theta_reader:
      theta = pickle.load(theta_reader)
  else:
    with open(theta_file, 'rb') as theta_reader:
      theta = pickle.load(theta_reader)


  with Writer(output_file) as writer:
    [writer.write('%20.8f\n' % v) for v in theta]
