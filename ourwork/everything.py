import numpy as np
import matplotlib.pyplot as plt
import csv
from pylab import figure, show, rand
from metric import analyze
import os
from sky import *
from collections import Counter
from machine_learning import *

def generate_halo_mag_data(fname, sky_range=None, scaling_factor=1 ):
  skies = objectify_data(test=False, sky_range=sky_range)
  c = csv.writer(open(fname, "wb")) #Now write the array to a csv file
  c.writerow([str('n_actual_halos'),str('mag_h1'),str( 'mag_h2'),str( 'mag_h3'),str( 'mag_h4')])
  
  for sky in skies:
    foo, bar, baz, m = sky.better_subtraction(scaling_factor=scaling_factor, for_training_data=True)
    row = [sky.n_halos()] + m
    c.writerow(row)
    print row


def objectify_data(sky_range=None):
  """ Creates a list of skies with all sky, galaxy, and all known halo 
      information.
      (Only contains halo information if reading from the training dataset)

      test: If True, loads test dataset. If False, loads training dataset.
      sky_range: List of which skies to load. Indicies match the actual
        sky number. E.g., sky_range=[210, 300] will load skies #210 and #300

      returns: list of skies
  """
  foo = raw_input("Type 'train' for training data, 'test' for test data")
  if foo == 'test':
    test = True
  else if foo = 'train':
    test = False
  else:
    print "Invalid entry"
    return
    
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
    skyid = "Sky%d" % k
    if test:
      sky = Sky(skyid=skyid)
      data_filename = '../data/Test_Skies/Test_Sky%i.csv' % k
    else:
      hs1 = Halo(x1[k-1], y1[k-1]) if (x1[k-1] != 0.0 or y1[k-1] != 0.0) else None
      hs2 = Halo(x2[k-1], y2[k-1]) if (x2[k-1] != 0.0 or y2[k-1] != 0.0) else None
      hs3 = Halo(x3[k-1], y3[k-1]) if (x3[k-1] != 0.0 or y3[k-1] != 0.0) else None
      sky = Sky(skyid=skyid, halo1=hs1, halo2=hs2, halo3=hs3)
      data_filename = '../data/Train_Skies/Training_Sky%i.csv' % k
    res.append(sky)

    x, y, e1, e2 = np.loadtxt(data_filename, delimiter=',', unpack=True, usecols=(1,2,3,4), skiprows=1)
    for i in range(len(x) - 1):
      sky.add_galaxy(Galaxy(x[i], y[i], e1[i], e2[i]))

  return res


def dist_between_halos(sky_range=None):
  skies = objectify_data(test=False, sky_range=sky_range)

  ## what is the average distance between 2 halos?
  ds_2halos = [s.actual[0].euclid_dist(s.actual[1]) for s in skies if s.n_halos() == 2]
  ds_3halos = [s.actual[0].euclid_dist(s.actual[1]) for s in skies if s.n_halos() == 3] + \
              [s.actual[1].euclid_dist(s.actual[2]) for s in skies if s.n_halos() == 3] + \
              [s.actual[0].euclid_dist(s.actual[2]) for s in skies if s.n_halos() == 3]

  counts, bin_edges = np.histogram(np.array(ds_3halos), bins=100, density=False)
  cdf1 = np.cumsum(counts) * 1.0/sum(counts)
  plt.plot(bin_edges[1:], cdf1)


  counts, bin_edges = np.histogram(np.array(ds_2halos), bins=100, density=False)
  cdf2 = np.cumsum(counts) * 1.0/sum(counts)
  plt.plot(bin_edges[1:], cdf2, color='red')
  plt.show()

  ## given the distance between 2 halos, how far is each of those from the 3rd?
  ds_0_to_1 = [s.actual[0].euclid_dist(s.actual[1]) for s in skies if s.n_halos() == 3]
  ds_1_to_2 = [s.actual[1].euclid_dist(s.actual[2]) for s in skies if s.n_halos() == 3]
  ds_0_to_2 = [s.actual[0].euclid_dist(s.actual[2]) for s in skies if s.n_halos() == 3]

  plt.scatter(ds_0_to_1, ds_1_to_2)
  plt.scatter(ds_0_to_1, ds_0_to_2)
  plt.scatter(ds_1_to_2, ds_0_to_1)
  plt.scatter(ds_1_to_2, ds_0_to_2)
  plt.scatter(ds_0_to_2, ds_0_to_1)
  plt.scatter(ds_0_to_2, ds_1_to_2)
  plt.show()

  ## what do the areas of triangles formed by 3 halos look like?
  areas = [np.abs(s.actual[0].x * (s.actual[1].y - s.actual[2].y) + \
                  s.actual[1].x * (s.actual[2].y - s.actual[0].y) + \
                  s.actual[2].x * (s.actual[0].y - s.actual[1].y) \
           ) / 2.0 for s in skies if s.n_halos() == 3]

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
    points, tqs = sky.non_binned_signal(nhalos=2)
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

def k_nearest_neighbor(training_data, test_point, k):
  #find the distance from the test_point (a vector of magnitudes) 
  # to each example in training_data
  map_of_stuff = []
  for t in training_data:
    diff = t.x - test_point.x
    dist = np.sqrt(np.sum(np.square(diff)))
    map_of_stuff.append((dist, t.y))

  #find the k-nearest examples to the test_point
  map_of_stuff.sort()
  votes = [y for dist, y in map_of_stuff]

  return Counter(votes[0:k]).most_common(1)[0][0]





def analyze_magnitude_of_halos():
  """
  Analayzes the file "predicted_mag_of_halos.csv".

  rows are in format: nhalos, m1, m2, m3, m4
  nhalos is the actual number of halos in the sky
  m(n) is the magnitude of the nth halo.

  hypothesis: If halo n exists, and halo n+1 does not, there will
  be a different relationship btwwn m(n) and m(n+1) than in the case
  where h(n) and h(n+1) both exist
  """
  
  fname = 'predicted_mag_of_halos.csv'
  read_file = csv.reader(open(fname, 'rb'))
  training_data = []
  cross_val_data = []
  
  #group data as training/cross_validation
  for i, row in enumerate(read_file):
    if row[0] == 'n_actual_halos': continue  #ignore header row
    
    y = float(row[0])
    base = float(row[1])
    # X = np.array([1.0, float(row[2])/base, float(row[3])/base, float(row[4])/base])
    X = np.array([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
    
    #divide by a high number to group everything as training data
    if (i % 4) == 0:
      cross_val_data.append(Datapoint(X, y))
    else:
      training_data.append(Datapoint(X, y))

  for k in range(1, 11):
    success = 0
    predictions = 0

    # for dp in cross_val_data:
    #   if dp.y != 1: continue
    #   predictions += 1
    #   pred = k_nearest_neighbor(training_data, dp, k)
    #   if pred == dp.y: 
    #     success += 1

    # print "k=%i: %i successful predictions out of %i guesses" % (k, success, predictions)


    # regroup data with X=[m1:m0, m2:m1, m3:m2, m2:m0, m3:m1, m3:m0]
    new_training_data, new_cross_data = [], []
    for ex in training_data:
      a = ex.x[1] / ex.x[0]
      b = ex.x[2] / ex.x[1]
      c = ex.x[3] / ex.x[2]

      d = ex.x[2] / ex.x[0]
      e = ex.x[3] / ex.x[1]

      f = ex.x[3] / ex.x[0]
      x = np.array([a, b, c, d, e, f])
      new_training_data.append(Datapoint(x, ex.y))
    
    for ex in cross_val_data:
      # if ex.y != 1: continue
      predictions += 1
      a = ex.x[1] / ex.x[0]
      b = ex.x[2] / ex.x[1]
      c = ex.x[3] / ex.x[2]

      d = ex.x[2] / ex.x[0]
      e = ex.x[3] / ex.x[1]

      f = ex.x[3] / ex.x[0]
      x = np.array([a, b, c, d, e, f])

      point = Datapoint(x, ex.y)
      pred = k_nearest_neighbor(new_training_data, point, k)
      if pred == point.y:
        success += 1
    print "k=%i: %i successful predictions out of %i guesses" % (k, success, predictions)
  
  # ratios_at_cutoff, ratios_past_cutoff = [], []
  # ratios_before_cutoff = []
  # for ex in training_data:
  #   if ex.y == 1:
  #     ratios_at_cutoff.append((ex.x[2] / ex.x[1]))
  #     ratios_past_cutoff.append((ex.x[2] / ex.x[1]))

  #   # if ex.y == 2:
  #   #   ratios_at_cutoff.append(())

  # counts, bin_edges = np.histogram(ratios_at_cutoff, bins=100, density=False)
  # cdf1 = np.cumsum(counts) * 1.0 / sum(counts)
  # plt.plot(bin_edges[1:], cdf1, color='red')

  # counts, bin_edges = np.histogram(ratios_past_cutoff, bins=100, density=False)
  # cdf2 = np.cumsum(counts) * 1.0 / sum(counts)
  # plt.plot(bin_edges[1:], cdf2, color='blue')

  # plt.show()


def write_data(skies=None, output_file='genericOutput.csv', method=None, opts={}):
  if skies == None:
    skies = objectify_data()

  print "Writing %s" % output_file
  c = csv.writer(open(output_file, "wb")) #Now write the array to a csv file
  c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
  
  for sky in skies:
    sky.predictions = method(sky, **opts)
    sky_output = sky.formatted_output_list()
    print sky_output
    c.writerow(sky_output)