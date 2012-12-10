import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse 
import copy
import numpy as np
from scipy.optimize import newton
from pylab import figure, show, rand
import random
from collections import Counter
from machine_learning import *

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def euclid_dist(self, other):
    return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Halo(Point):
  def __str__(self):
    return "x=%.1f, y=%.1f" % (self.x, self.y)


class Universe:
  def __init__(self, sky_range=None):
    self.skies = self.objectify_data(sky_range)


  def overguess_positions_of_halos(self):
    """ Predict the position of 5 halos in every sky, and write that
    data to a csv file. Can be used for later analyses, including predicting
    the number of actual halos in the sky, and trimming the data accordingly.

    NOTE: Very slow. Should only be run if the halo prediction method is changed.
    Otherwise, load data via self.__load_saved_data()
    """

    output_file = self.__saved_positions_and_magnitudes()
    print "Writing %s" % output_file
    
    #write 5 halos' positions and their respective magnitudes to a csv file
    c = csv.writer(open(output_file, "wb")) #Now write the array to a csv file
    c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),\
      str( 'pred_x2'), str( 'pred_y2'), str( 'pred_x3'), str(' pred_y3'),\
       str('pred_x4'), str('pred_y4'), str('pred_x5'), str('pred_y5'), \
       str('m1'), str('m2'), str('m3'), str('m4'), str('m5')])
    
    for sky in self.skies:
      halos, foo, bar, ms = sky.better_subtraction()
      sky.predictions = halos
      sky_output = sky.formatted_output_list() + ms
      print "Writing 5 halo postions + magnitudes for %s" % sky.skyid
      c.writerow(sky_output)

  def write_predictions_to_file(self, output_file):
    training_data = objectify_training_data('training_data_s=10.csv')
    saved_data = self.__load_saved_data()

    c = csv.writer(open(output_file, "wb"))
    c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
  
    for i, sky in enumerate(self.skies):
      ms = saved_data[i][11:]
      pred_halos = sky.predictions

      #Assume that halos at the edge of a sky are incorrect, and that any halos
      #after them are also incorrect
      if (pred_halos[1].x == 0 or pred_halos[1].y == 0):
        print "chopped 2 halos"
        nhalos = 1
      elif pred_halos[2].x == 0 or pred_halos[2].y == 0:
        print "chopped 1 halo"
        nhalos = 2
      
      #Otherwise, use our semi-accurate k-nearest neighbors method
      else:
        nhalos = sky.predict_number_of_halos(ms, training_data=training_data)

      sky.remove_non_existent_halos(pred_halos, nhalos)
      sky_output = sky.formatted_output_list()
      c.writerow(sky_output)


  def __load_saved_data(self):
    """Loads the data in TRAIN_predicted_position_of_5halos.csv or
    TEST_predicted_position_of_5halos.csv.

    """
    fname = self.__saved_predictions_filename()
    reader = csv.reader(open(fname, 'rb'))

    data = []
    for row in reader:
      if row[0] == 'SkyId': continue
      new_row = [row[0]] + [float(cell) for cell in row[1:]]
      data.append(new_row)

    return data

  def __saved_predictions_filename(self):
    if self.test:
      return 'TEST_predicted_position_of_5halos.csv'
    else: 
      return 'TRAIN_predicted_position_of_5halos.csv'

  def objectify_data(self, sky_range=None):
    """ Creates a list of skies with all sky, galaxy, and all known halo 
        information.
        (Only contains halo information if reading from the training dataset)

        test: If True, loads test dataset. If False, loads training dataset.
        sky_range: List of which skies to load. Indicies match the actual
          sky number. E.g., sky_range=[210, 300] will load skies #210 and #300

        returns: list of skies
    """
    #Ask the user whether to use test or training data.
    foo = raw_input("Type 'train' for training data, 'test' for test data:\n")
    if foo == 'test': self.test = True
    elif foo == 'train': self.test = False
    else: return
      
    if self.test:
      n_skies_filename = '../data/Test_haloCounts.csv'
    else:
      n_skies_filename = '../data/Training_halos.csv'
      x1, y1, x2, y2, x3, y3 = np.loadtxt('../data/Training_halos.csv', delimiter=',', unpack=True, usecols=(4,5,6,7,8,9), skiprows=1)
  
    saved_data = self.__load_saved_data()
  
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
      if self.test:
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


      pred_h1_x, pred_h1_y, pred_h2_x, pred_h2_y, pred_h3_x, pred_h3_y = saved_data[k-1][1:7]
      sky.predictions = [Halo(pred_h1_x, pred_h1_y), Halo(pred_h2_x, pred_h2_y), Halo(pred_h3_x, pred_h3_y)]
      print "Loaded %s" % skyid

    return res



class Galaxy(Point):
  """
  x, y represent the position of the galaxy
  e1, e2 represent its ellipticity, as defined by Kaggle
  a, b are proportional to e1/e2, and necessary for plotting galaxies
  """
  def __init__(self, x, y, e1, e2):
    self.x = x
    self.y = y
    self.e1 = e1
    self.e2 = e2
  
  def a(self):
    theta = np.arctan(self.e2/self.e1) / 2
    self.theta = theta * 180 / np.pi

    E = self.e1 / np.cos(2 * theta)
    return 50/(1-E)

  def b(self):
    theta = np.arctan(self.e2/self.e1) / 2
    self.theta = theta * 180 / np.pi

    E = self.e1 / np.cos(2 * theta)
    return 50/(1 + E)


class Sky:
  """
  skyid: string, as represented in the given data
  galaxies: a list of Galaxies, as represented in the given data
  predictions: a list of Halos (up to 3), representing predicted halo positions
  actual: a list of Halos (up to 3), representing known halo positions. [Applicable
  only to training data]
  """
  def __init__(self, skyid, halo1=None, halo2=None, halo3=None):
    self.skyid = skyid
    self.galaxies = []
    self.n_halos_val = None
    self.predictions = [None, None, None]
    self.actual = [halo1, halo2, halo3]
 
  def add_galaxy(self, galaxy):
    """
    Appends a Galaxy to the sky's list of galaxies
    """
    self.galaxies.append(galaxy)

  def formatted_output_list(self):
    """
    Returns list of predicted halos: [h1_x, h1_y, h2_x, h2_y, h3_x, h3_y]
    If a halo doesn't exist, it replaces "None" with the point (0, 0)
    Used for formatting output to submit to Kaggle
    """
    output_list = [self.skyid]
    for halo in self.predictions:
      output_list.append(halo.x if halo else 0.0)
      output_list.append(halo.y if halo else 0.0)
    return output_list
  
  def n_halos(self):
    """Number of actual halos. [Applicable only to training data]"""
    return sum([1 for h in self.actual if h != None])


  def plot_galaxies(self, ax, gals=None):
    if gals==None:
      gals = self.galaxies
    for gal in gals:
      e = Ellipse((gal.x, gal.y), gal.a(), gal.b(), angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)

  def plot(self):
    """Plots the predicted signals caused by each halo, as well as a cumulative signal.
    Also plots the galaxies in the sky, with their ellipticities as caused by the halos."""
    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    x_rs, y_rs = np.meshgrid(x_r_range, y_r_range)
    
    halos, tqs, orig_galaxies, ms = self.better_subtraction()
    halo1, halo2, halo3 = halos
    tq1, tq2, tq3 = tqs
    gs1, gs2, gs3 = orig_galaxies

    fig = figure(figsize=(11,11)) 
    for tq, subplotid, title, gal in [(tq1, 221, 'Signal 1', gs1), (tq2, 222, 'Signal 2', gs2), (tq3, 223, 'Signal 3', gs3)]:
      
      #plot map of signal
      if tq != None: 
        ax = fig.add_subplot(subplotid, aspect='equal')
        plt.title("%s: %s" % (self.skyid, title))

        tq = np.array(tq).reshape((len(x_r_range),len(y_r_range)))
        plt.contourf(x_rs, y_rs, tq, 20)
        plt.colorbar()
        plt.clim(-0.1, 0.1)

        self.plot_galaxies(ax,gal)

        for i, color in enumerate(['black', 'blue', 'pink']):
          if self.actual[i] != None:
            plt.plot(self.actual[i].x, self.actual[i].y, marker='*', markersize=20, color=color)

        if halo1: plt.plot(halo1.x, halo1.y, marker='o', markersize=10, color='black')
        if halo2: plt.plot(halo2.x, halo2.y, marker='o', markersize=10, color='blue')
        if halo3: plt.plot(halo3.x, halo3.y, marker='o', markersize=10, color='pink')
    show()
 
    return halos

  def better_subtraction(self, training_data=None, to_file=False, for_training_data=False):
    """
    Predicts the position of up to 5 halos in the sky. Note that we're 
    overpredicting, since there can't be more than 3 halos. This extra info is
    used for training data, and helped us test several hypotheses.

    Returns:
    halos: list of 5 predicted Halos
    signal_maps: list of 5 signal maps, as caused by each halo
    orig_galaxies: list of 3 sublists, where each sublist contains galaxies whose ellipticies
                  do not account for the effect of previous halos.

    ms: predicted magnitude of each halo. Magnitudes are calculated as the factor needed
        to completely negate a halo at its location. eg, if a halo is predicted to be at 
        (100, 200), we find the value of m that forced the signal at (100, 200) to be 0.

    """
    nhalos = 5

    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    halos = []
    signal_maps = []
    orig_galaxies = []
    ms = []


    # Will be modifying galaxies in sky, make copy to avoid modifying orig data
    selfcopy = copy.deepcopy(self) 
    
    # Predict location of each halo
    for i in range(nhalos):
      orig_galaxies.append(copy.deepcopy(selfcopy.galaxies))
      signals = []
      max_e = 0.0
      pred_x = 0.0
      pred_y = 0.0

      # Find x_r, y_r with the maximum signal - make that the guess for the halo
      for y_r in y_r_range:
        for x_r in x_r_range:
          # See if x_r, y_r is the position of another halo
          on_other_halo = False
          for halo in halos:
            if halo.x == x_r and halo.y == y_r:
              on_other_halo = True

          # Find signal at x_r, y_r (if not on the position of another halo)
          if on_other_halo:
            signals.append(0.0)
          else:
            signal = selfcopy.mean_signal(x_r, y_r) 
            signals.append(signal)

            #if this is the greatest signal found so far, it's the best guess for halo's position
            if max_e < signal:
              max_e = signal
              pred_x = x_r
              pred_y = y_r

      #keep track of the halo, and the signal map
      new_halo = Halo(x=pred_x, y=pred_y)
      halos.append(new_halo)
      signal_maps.append(signals)

      #find proposed halo
      m = newton(selfcopy.mean_of_tangential_force_at_given_halo, 100, args=(new_halo,))
      ms.append(m)
      selfcopy.remove_effect_of_halo(m, new_halo, selfcopy)

    return halos, signal_maps, orig_galaxies[0:3], ms


  def remove_non_existent_halos(self, halos, pred_nhalos):
    """We always assume there are 5 halos, then make an educated guess at the 
    real number of halos. This function sets our prediction, based on the
    predicted number of halos.
    """

    while len(halos) > pred_nhalos:
      del halos[-1]

    for i in range(3-pred_nhalos):
      halos.append(None)

    self.predictions = halos
    

  def predict_number_of_halos(self, ms, training_data=None):
    """ Creates a feature vector of 10 features, based on the 5 halos' 
    magnitudes and their ratios to each other. Uses the k-nearest neighbor
    algorithm to predict the number of halos in this sky. 

    Note: Not very accurate. (Approx 50 percent correct when it has 6 or 7
          votes for a certain number of halos. Otherwise, 33 pecent correct)
    """

    if training_data == None:
      training_data = objectify_training_data()
    
    #create feature vector for sky
    x1, x2, x3, x4, x5 = ms
    ratios = [x2/x1, x3/x2, x4/x3, x5/x4, \
                x3/x1, x4/x2, x5/x3, \
                x4/x1, x5/x2, \
                x5/x1]
    X = np.array(ratios)

    #distance from self to each training example
    map_of_distances = []
    for t in training_data:
      diff = t.x - X
      dist = np.sqrt(np.sum(np.square(diff)))
      map_of_distances.append((dist, t.y))

    #find the 7 nearest examples to self
    map_of_distances.sort()
    votes = [y for dist, y in map_of_distances]

    return int(Counter(votes[0:7]).most_common(1)[0][0])

  def mean_of_tangential_force_at_given_halo(self, m, halo):
    """
      self: ideal sky with perfectly circular galaxies.
      m: magnitude of the effect of the halo on other galaxies
    """
    copy_of_sky = copy.deepcopy(self)
    copy_of_sky.remove_effect_of_halo(m, halo, self)
    return copy_of_sky.mean_signal(halo.x, halo.y)

  def remove_effect_of_halo(self, m, halo, original_sky):
    """
    Cancels out the effect of a halo(X_h, Y_h) on all galaxies in self.

    If self is an ideal sky, self is modified.
    If self is equal to original_sky, original_sky is modified.

    Whatever is modified is equal to original_sky, minus the effects of the halo.
    """
    for i, gal in enumerate(self.galaxies):
      #phi = angle btwn x axis and the line from the halo to the galaxy
      phi = np.arctan((gal.y - halo.y) / (gal.x - halo.x)) 
      
      #theta = angle between x axis and the galaxy's major axis
      theta = phi - np.pi / 2 
      r = gal.euclid_dist(halo)

      #calculate ellipticity added to the ideal_sky, given Halo and m
      e1 = m / (- r * ((np.tan(2 * theta) * np.sin(2 * phi)) + np.cos(2 * phi)))
      e2 = e1 * np.tan(2 * theta)

      #modify ideal_sky, so it is a copy of original_sky, less the proposed effects of a halo
      self.galaxies[i].e1 = original_sky.galaxies[i].e1 - e1
      self.galaxies[i].e2 = original_sky.galaxies[i].e2 - e2

  def mean_signal(self, x_prime, y_prime):
    """
    Calculates the mean signal in a sky at a given(x_prime,y_prime).
    """
    return np.mean(self.__signal__(x_prime, y_prime))


  def __signal__(self, x_prime, y_prime):
    """
    Ellipticity of each galaxy that is tangential to (x_prime, y_prime).

    As defined: http://www.kaggle.com/c/DarkWorlds/details/an-introduction-to-ellipticity
     - "the force exerted by the dark matter halo on the galaxy is tangential."
    
    So, we are assuming that all of the tangential ellipticity is a result of 
    the force exerted by the dark matter halo. 

    NOTE: If we have 1 halo, this assumption is true. If we have >1 halo, isn't 
          it less true?
    """
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])
    
    phi = np.arctan((y - y_prime)/(x - x_prime))
    e_tang = -(e1 * np.cos(2 * phi) + e2 * np.sin(2 * phi)) 
    return e_tang 
