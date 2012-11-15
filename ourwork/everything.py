import numpy as np
import csv
from pylab import figure, show, rand
from metric import analyze
import os
from sky import *


def file_len(fname):
  """ Calculate the length of a file
  Arguments:
         Filename: Name of the file wanting to count the rows of
  Returns:
         i+1: Number of lines in file
  """

  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def objectify_data(test=True, sky_range=None):
  if test:
    n_skies=file_len('../data/Test_haloCounts.csv')-1 #The number of skies in total
  else:
    n_skies=file_len('../data/Training_halos.csv')-1
    x1, y1, x2, y2, x3, y3 = np.loadtxt('../data/Training_halos.csv', \
                                        delimiter=',', unpack=True, usecols=(4,5,6,7,8,9),skiprows=1)
  
  if sky_range == None:
    sky_range = range(n_skies)

  res = []
  
  for k in sky_range:
    if test:
      sky = Sky()
    else:
      sky = Sky((x1[k],y1[k]), (x2[k],y2[k]), (x3[k], y3[k]))
    p=k+1
    
    if test:
      x,y,e1,e2=np.loadtxt('../data/Test_Skies/Test_Sky%i.csv' % p,\
             delimiter=',',unpack=True,usecols=(1,2,3,4),skiprows=1)
    else:
      x,y,e1,e2=np.loadtxt('../data/Train_Skies/Training_Sky%i.csv' % p,\
             delimiter=',',unpack=True,usecols=(1,2,3,4),skiprows=1)
    
    for i in range(len(x) - 1):
      sky.add_galaxy(Galaxy(x[i], y[i], e1[i], e2[i]))
    res.append(sky)
  return res

def optimize_bins(min_bins=1, max_bins=30):
  """
  Iterates over different values of nbins to find the optimum
  number of bins for the gridded_signal method.
  """
  skies = objectify_data(test=False)

  results = {}
  for n in range(min_bins, max_bins):
    output_file = "optimize_bins_%ibins.csv" % n
    
    write_data(skies, output_file, Sky.gridded_signal, {"nbin": n})
    metric = analyze(output_file, '../data/Training_halos.csv')
    results[n] = metric

    os.remove(output_file)

  for k, v in results.items():
    print k, v

def optimize_radius(min_radius = 200, max_radius = 800):
  """
  Iterates over different values of nbins to find the optimum
  number of bins for the gridded_signal method.
  """
  skies = objectify_data(test=False)

  results = {}
  for n in range(min_radius, max_radius, 100):
    output_file = "optimize_radius_%ir.csv" % n
    
    write_data(skies, output_file, Sky.gridded_signal, {"radius": n})
    metric = analyze(output_file, '../data/Training_halos.csv')
    results[n] = metric

    os.remove(output_file)

  for k, v in results.items():
    print k, v

def write_data(skies=None, output_file='genericOutput.csv', method=None, opts={}):
  if skies == None:
    skies = objectify_data()

  print "Writing %s" % output_file
  c = csv.writer(open(output_file, "wb")) #Now write the array to a csv file
  c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
  
  for k in xrange(len(skies)):
    halostr=['Sky'+str(k+1)]
    # pos_halo= skies[k].gridded_signal()
    # pos_halo = skies[k].max_likelihood()
    pos_halo = method(skies[k], **opts)
    for n in xrange(3):
      halostr.append(pos_halo[n][0])
      halostr.append(pos_halo[n][1])

    c.writerow(halostr)