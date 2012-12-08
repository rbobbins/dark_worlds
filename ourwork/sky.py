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
  def __init__(self, skies, test=False):
    self.test = test
    self.skies = skies


  def overguess_positions_of_halos(self):
    """ Predict the position of 5 halos in every sky, and write that
    data to a csv file. Can be used for later analyses, including predicting
    the number of actual halos in the sky, and trimming the data accordingly.
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
      pred_halos = [Halo(saved_data[i][1], saved_data[i][2]), Halo(saved_data[i][3], saved_data[i][4]), Halo(saved_data[i][5], saved_data[i][6])]
      
      #First, do predictions for skies with edge halos (delete those halos, + anything past them)
      if (pred_halos[1].x == 0 or pred_halos[1].x == 4130 or pred_halos[1].y == 0 or pred_halos[1].y == 4130):
        print "chopped 2 halos"
        nhalos = 1
      elif pred_halos[2].x == 0 or pred_halos[2].x == 4130 or pred_halos[2].y == 0 or pred_halos[2].y == 4130:
        print "chopped 1 halo"
        nhalos = 2
      #Then, use our k-nearest neighbors method
      else:
        votes = sky.predict_number_of_halos(ms, training_data=training_data)[1:]
        nhalos = int(Counter(votes).most_common(1)[0][0])

      sky.remove_non_existent_halos(pred_halos, nhalos)
      sky_output = sky.formatted_output_list()
      c.writerow(sky_output)


  def __load_saved_data(self):
    fname = self.__saved_positions_and_magnitudes()
    reader = csv.reader(open(fname, 'rb'))

    data = []
    for row in reader:
      if row[0] == 'SkyId': continue
      new_row = [row[0]] + [float(cell) for cell in row[1:]]
      data.append(new_row)

    return data

  def __saved_positions_and_magnitudes(self):
    if self.test:
      return 'TEST_predicted_position_of_5halos.csv'
    else: 
      return 'TRAIN_predicted_position_of_5halos.csv'



class Galaxy(Point):
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
  def __init__(self, skyid, halo1=None, halo2=None, halo3=None):
    self.skyid = skyid
    self.galaxies = []
    self.n_halos_val = None
    self.predictions = [None, None, None]
    self.actual = [halo1, halo2, halo3]
  
  def formatted_output_list(self):
    output_list = [self.skyid]
    for halo in self.predictions:
      output_list.append(halo.x if halo else 0.0)
      output_list.append(halo.y if halo else 0.0)
    return output_list
  
  def n_halos(self):
    if not self.n_halos_val:
      self.n_halos_val = sum([1 for h in self.actual if h != None])
    return self.n_halos_val

  def add_galaxy(self, galaxy):
    self.galaxies.append(galaxy)

  def plot_galaxies(self, ax, gals=None):
    if gals==None:
      gals = self.galaxies
    for gal in gals:
      e = Ellipse((gal.x, gal.y), gal.a(), gal.b(), angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)

  def plot(self):
    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    x_rs, y_rs = np.meshgrid(x_r_range, y_r_range)
    
    halos, tqs, orig_galaxies, ms = self.better_subtraction()
    halo1, halo2, halo3 = halos
    tq1, tq2, tq3 = tqs
    gs1, gs2, gs3 = orig_galaxies
    # total_signal = np.array(tq1) + np.array(tq2) + np.array(tq3)

    fig = figure(figsize=(11,11)) 
    for tq, subplotid, title, gal in [(tq1, 221, 'Signal 1', gs1), (tq2, 222, 'Signal 2', gs2), (tq3, 223, 'Signal 3', gs3)]:
      
      #plot map of signal
      if tq != None: 
        ax = fig.add_subplot(subplotid, aspect='equal')
        plt.title("%s: %s" % (self.skyid, title))

        tq = np.array(tq).reshape((len(x_r_range),len(y_r_range)))
        # total_signal += tq
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

  def better_subtraction(self, training_data=None, to_file=False, for_training_data=False, scaling_factor=1):
    nhalos = 5

    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    halos = []
    signal_maps = []
    orig_galaxies = []
    ms = []

    # Predict location of each halo
    selfcopy = copy.deepcopy(self) # Will be modifying galaxies in sky, doing this not to modify orig data
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
      selfcopy.remove_effect_of_halo((m*scaling_factor), new_halo, selfcopy)

    # if not for_training_data:
    #   #find the number of halos that *actually* exist
    # actual_nhalos = self.predict_number_of_halos(ms, training_data)
    #   #delete extraneous data
    #   while len(halos) > actual_nhalos:
    #     del halos[-1]
    #     del signal_maps[-1]
    #     del ms[-1]

    #   for i in range(3-actual_nhalos):
    #     halos.append(None)
    #     signal_maps.append(None)

    # # Return values
    # if to_file:
    #   return halos
    # else:
    return halos, signal_maps, orig_galaxies[0:3], ms


  def remove_non_existent_halos(self, halos, pred_nhalos):
    while len(halos) > pred_nhalos:
      del halos[-1]
      # del signal_maps[-1]
      # del ms[-1]

    for i in range(3-pred_nhalos):
      halos.append(None)
      # signal_maps.append(None)

    self.predictions = halos
    

  def predict_number_of_halos(self, ms, training_data=None):
    if training_data == None:
      training_data = objectify_training_data()
    
    x1, x2, x3, x4, x5 = ms
    ratios = [x2/x1, x3/x2, x4/x3, x5/x4, \
                x3/x1, x4/x2, x5/x3, \
                x4/x1, x5/x2, \
                x5/x1]
    X = np.array(ratios)

    map_of_distances = []
    for t in training_data:
      diff = t.x - X
      dist = np.sqrt(np.sum(np.square(diff)))
      map_of_distances.append((dist, t.y))

    #find the k-nearest examples to X
    map_of_distances.sort()
    votes = [y for dist, y in map_of_distances]

    return votes[0:8]
    # return int(Counter(votes[0:7]).most_common(1)[0][0])

  def mean_of_tangential_force_at_given_halo(self, m, halo):
    """
      self: ideal sky with perfectly circular galaxies.
      m: magnitude of the effect of the halo on other galaxies
    """
    copy_of_sky = copy.deepcopy(self)
    copy_of_sky.remove_effect_of_halo(m, halo, self)
    return copy_of_sky.sum_signal(halo.x, halo.y)

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


  def sum_signal(self, x_prime, y_prime):
    """
    Calculates the sum of the signal in a sky at a given(x_prime, y_prime)
    """
    return np.sum(self.__signal__(x_prime, y_prime))

  def __signal__(self, x_prime, y_prime):
    """
    Ellipticity of each galaxy that is tangential to (x_prime, y_prime).

    As defined: http://www.kaggle.com/c/DarkWorlds/details/an-introduction-to-ellipticity
     - "the force exerted by the dark matter halo on the galaxy is tangential."
    
    So, we are assuming that all of the tangential ellipticity is a result of the
    force exerted by the dark matter halo. 

    NOTE: If we have 1 halo, this is true. If we have >1 halo, isn't this less true?
    """
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])
    
    phi = np.arctan((y - y_prime)/(x - x_prime))
    e_tang = -(e1 * np.cos(2 * phi) + e2 * np.sin(2 * phi)) 
    return e_tang 
