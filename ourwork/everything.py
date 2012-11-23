import numpy as np
import matplotlib.pyplot as plt
import csv
from pylab import figure, show, rand
from metric import analyze
import os
from sky import *


def objectify_data(test=True, sky_range=None):
  """ Creates a list of skies with all sky, galaxy, and all known halo 
      information.
      (Only contains halo information if reading from the training dataset)

      test: If True, loads test dataset. If False, loads training dataset.
      sky_range: List of which skies to load. Indicies match the actual
        sky number. E.g., sky_range=[210, 300] will load skies #210 and #300

      returns: list of skies
  """

  if test:
    n_skies_filename = '../data/Test_haloCounts.csv'
  else:
    n_skies_filename = '../data/Training_halos.csv'
    x1, y1, x2, y2, x3, y3 = np.loadtxt('../data/Training_halos.csv', delimiter=',', unpack=True, usecols=(4,5,6,7,8,9), skiprows=1)
  if sky_range == None:
    # Find the length of the file
    with open(n_skies_filename) as f:
      for i, l in enumerate(f):
        pass
    # Make sky_range cover 1 through the number of skies in the file
    sky_range = range(1, i+1)

  res = []
  for k in sky_range:
    if test:
      sky = Sky()
      data_filename = '../data/Test_Skies/Test_Sky%i.csv' % k
    else:
      hs = [(x1[k-1], y1[k-1]), None, None]
      if (x2[k-1] != 0.0 or y2[k-1] != 0.0):
        hs[1] = (x2[k-1], y2[k-1])
      if x3[k-1] != 0.0 or y3[k-1] != 0.0:
        hs[2] = (x3[k-1], y3[k-1])
      sky = Sky(*hs)
      data_filename = '../data/Train_Skies/Training_Sky%i.csv' % k
    
    x, y, e1, e2 = np.loadtxt(data_filename, delimiter=',', unpack=True, usecols=(1,2,3,4), skiprows=1)
    res.append(sky)
    for i in range(len(x) - 1):
      sky.add_galaxy(Galaxy(x[i], y[i], e1[i], e2[i]))

  return res

def euclidean_distance(point1, point2):
  return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def dist_between_halos(sky_range=None):
  skies = objectify_data(test=False, sky_range=sky_range)

  ## what is the average distance between 2 halos?
  ds_2halos = [euclidean_distance(s.actual[0], s.actual[1]) for s in skies if s.n_halos() == 2]

  ds_3halos = [euclidean_distance(s.actual[0], s.actual[1]) for s in skies if s.n_halos() == 3] + \
              [euclidean_distance(s.actual[1], s.actual[2]) for s in skies if s.n_halos() == 3] + \
              [euclidean_distance(s.actual[0], s.actual[2]) for s in skies if s.n_halos() == 3]


  counts, bin_edges = np.histogram(np.array(ds_3halos), bins=100, density=False)
  cdf1 = np.cumsum(counts) * 1.0/sum(counts)
  plt.plot(bin_edges[1:], cdf1)


  counts, bin_edges = np.histogram(np.array(ds_2halos), bins=100, density=False)
  cdf2 = np.cumsum(counts) * 1.0/sum(counts)
  plt.plot(bin_edges[1:], cdf2, color='red')
  plt.show()

  ## given the distance between 2 halos, how far is each of those from the 3rd?
  ds_0_to_1 = [euclidean_distance(s.actual[0], s.actual[1]) for s in skies if s.n_halos() == 3]
  ds_1_to_2 = [euclidean_distance(s.actual[1], s.actual[2]) for s in skies if s.n_halos() == 3]
  ds_0_to_2 = [euclidean_distance(s.actual[0], s.actual[2]) for s in skies if s.n_halos() == 3]

  plt.scatter(ds_0_to_1, ds_1_to_2)
  plt.scatter(ds_0_to_1, ds_0_to_2)
  plt.scatter(ds_1_to_2, ds_0_to_1)
  plt.scatter(ds_1_to_2, ds_0_to_2)
  plt.scatter(ds_0_to_2, ds_0_to_1)
  plt.scatter(ds_0_to_2, ds_1_to_2)
  plt.show()

  ## what do the areas of triangles formed by 3 halos look like?
  areas = [np.abs(s.actual[0][0] * (s.actual[1][1] - s.actual[2][1]) + \
           s.actual[1][0] * (s.actual[2][1] - s.actual[0][1]) + \
           s.actual[2][0] * (s.actual[0][1] - s.actual[1][1])) / 2.0 for s in skies if s.n_halos() == 3]

  counts, bin_edges = np.histogram(np.array(areas), bins=100, density =False)
  cdf = np.cumsum(counts) * 1.0 / sum(counts)
  plt.plot(bin_edges[1:], cdf)
  plt.show()


def analyze_ratio(sky_range=None):
  """
  TO USE: Comment out the line in sky.py > Sky.non_binned_signal
  which says "if self.actual[1] != None:"

  Analyzes the ratio between the predicted first signal, 
  and predicted second signal. 

  Thought it could be used to determine the number of halos in
  a sky. Not possible, because the results were not expected

  Hypothesis: If there's only 1 halo, there will be an order
  of magnitude difference btwn the signals. If there are 2 halos,
  the predicted signals will be within an order of magnitude.

  Result: This was not the case. If there's only 1 actual halo, it's
  second signal will be close to the first
  """
  skies = objectify_data(test=False, sky_range=sky_range)
  ratios1, ratios2 = [], []
  for sky in skies:
    points, tqs = sky.non_binned_signal()
    sig1 = max(tqs[0]) if tqs[0] else 0.0
    sig2 = max(tqs[1]) if tqs[1] else 0.0
    ratio = (sig2 * 1.0/sig1)
    print ratio
    if sky.n_halos() == 1:
      ratios1.append(ratio)
    elif sky.n_halos() == 2:
      ratios2.append(ratio)
  
  counts, bin_edges = np.histogram(np.array(ratios1), bins=100, density=False)
  cdf1 = np.cumsum(counts) * 1.0 / sum(counts)

  counts, bin_edges = np.histogram(np.array(ratios2), bins=100, density=False)
  cdf2 = np.cumsum(counts) * 1.0 / sum(counts)
  
  plt.plot(bin_edges[1:], cdf1, color='red')
  plt.plot(bin_edges[1:], cdf2, color='blue')
  plt.show()

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
      if pos_halo[n] == None:
        halostr.append('0.0')
        halostr.append('0.0')
      else:
        halostr.append(pos_halo[n][0])
        halostr.append(pos_halo[n][1])

    c.writerow(halostr)