import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse 
import copy
import numpy as np
from scipy.optimize import newton
from pylab import figure, show, rand
import random


class Halo:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def euclid_dist_from_halo(self, other):
    return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

  def __str__(self):
    return "x=%.1f, y=%.1f, sig=%.1f" % (self.x, self.y, self.signal)

class Galaxy:
  def __init__(self, x, y, e1, e2):
    self.x = x
    self.y = y
    self.e1 = e1
    self.e2 = e2
    # self.determine_axes()

  def euclid_dist_from_point(self, other):
    return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
  
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
  # def determine_axes(self):
  #   theta = np.arctan(self.e2/self.e1) / 2
  #   self.theta = theta * 180 / np.pi

  #   E = self.e1 / np.cos(2 * theta)
  #   self.a = 50/(1-E)
  #   self.b = 50/(1 + E)


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

  def non_binned_signal(self, to_file=False, nhalos=0):
    if nhalos == 0:
      nhalos = random.choice([1,2,3])
    nhalos = 3

    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    halos = []
    signal_maps = []

    # Predict location of each halo
    for i in range(nhalos):
      tq = []
      max_e = 0.0
      pred_x = 0.0
      pred_y = 0.0

      # Find x_r, y_r with the maximum signal - make that the guess for the halo
      for y_r in y_r_range:
        for x_r in x_r_range:
          # See if x_r, y_r is the position of another halo
          on_other_halo = False
          for halo in halos:
            if halo != None and halo.x == x_r and halo.y == y_r:
              done = True
              break

          # Find signal at x_r, y_r (if not on the position of another halo)
          if on_other_halo:
            tq.append(0.0)
          else:
            t = self.e_tang(x_r, y_r, halos) 
            tq.append(t)
            if max_e<t:
              max_e = t
              pred_x = x_r
              pred_y = y_r

      halos.append(Halo(x=pred_x, y=pred_y))
      signal_maps.append(tq)

    for i in range(3-nhalos):
      halos.append(None)
      signal_maps.append(None)

    # Return values
    if to_file == True:
      return halos
    else:
      return halos, signal_maps


  def plot(self):
    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    x_rs, y_rs = np.meshgrid(x_r_range, y_r_range)
    
    halos, tqs, orig_galaxies = self.better_subtraction()
    halo1, halo2, halo3 = halos
    tq1, tq2, tq3 = tqs
    gs1, gs2, gs3 = orig_galaxies


    total_signal = np.zeros((len(x_r_range),len(y_r_range)))
    fig = figure(figsize=(11,11)) 
    total_signal = np.array(tq1) + np.array(tq2)
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
     
    ax = fig.add_subplot(224, aspect='equal')
    plt.title("%s: total signal" % (self.skyid))
    total_signal = total_signal.reshape((len(x_r_range), y_r_range))
    print total_signal
    plt.contourf(x_rs, y_rs, total_signal, 20)
    plt.colorbar()
    plt.clim(-0.1, 0.1)
    show()
    return halos

  def better_subtraction(self):
    nhalos = 3

    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    halos = []
    signal_maps = []
    orig_galaxies = []

    # Predict location of each halo
    for i in range(nhalos):
      orig_galaxies.append(self.galaxies)
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
            signal = self.mean_signal(x_r, y_r) 
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
       #this gets overwritten on every iteration
      print "about to optimize x"
      x = newton(self.mean_of_tangential_force_at_given_halo, 100, args=(new_halo,))
      # print ideal_sky.sum_of_tangential_force_at_given_halo(x, halos[i], self) 
      print "optimized x as %f" % x
      self.remove_effect_of_halo(x, new_halo, self)


    for i in range(3-nhalos):
      halos.append(None)
      signal_maps.append(None)

    # Return values
    return halos, signal_maps, orig_galaxies


  def mean_of_tangential_force_at_given_halo(self, X, halo):
    """
      self: ideal sky with perfectly circular galaxies.
    """
    ideal_sky = self.idealized_copy()
    copy_of_sky = copy.deepcopy(self)
    #do this for the ideal sky
    copy_of_sky.remove_effect_of_halo(X, halo, self)
    # print ideal_sky.mean_signal(halo.x, halo.y)
    return copy_of_sky.sum_signal(halo.x, halo.y)

  def mean_signal(self, x_prime, y_prime):
    """
    Calculates the mean signal in a sky at a given(x_prime,y_prime).
    """
    return np.mean(self.__signal__(x_prime, y_prime))


  def sum_signal(self, x_prime, y_prime):
    return np.sum(self.__signal__(x_prime, y_prime))

  def __signal__(self, x_prime, y_prime):
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])
    
    phi = np.arctan((y - y_prime)/(x - x_prime))
    e_tang = -(e1 * np.cos(2 * phi) + e2 * np.sin(2 * phi)) 
    return e_tang   

  def remove_effect_of_halo(self, X, halo, original_sky):
    """
    Cancels out the effect of a halo(X_h, Y_h) on all galaxies in self.

    If self is an ideal sky, self is modified.
    If self is equal to original_sky, original_sky is modified.

    Whatever is modified is equal to original_sky, minus the effects of the halo.
    """
    for i, gal in enumerate(self.galaxies):
      phi = np.arctan((gal.y - halo.y) / (gal.x - halo.x)) #angle btwn x axis and the line from the halo to the galaxy
      theta = np.pi / 2 - phi
      r = gal.euclid_dist_from_point(halo)

      #calculate ellipticity added to the ideal_sky, given Halo and X
      e1 = X / (- r * ((np.tan(2 * theta) * np.sin(2 * phi)) + np.cos(2 * phi)))
      e2 = e1 * np.tan(2 * theta)

      #modify ideal_sky, so it is a copy of original_sky, less the proposed effects of a halo
      self.galaxies[i].e1 = original_sky.galaxies[i].e1 - e1
      self.galaxies[i].e2 = original_sky.galaxies[i].e2 - e2

  def idealized_copy(self):
    """
    Creates copy of self, with all e1 and e2 = 0.
    """
    dup = copy.deepcopy(self)

    for gal in dup.galaxies:
      gal.e1, gal.e2 = 0, 0

    return dup

  # def e_tang(self, pred_x, pred_y, other_halos=[]):
    # e_tang = self.total_signal()

    # if len(other_halos) > 0:
    #   vect_to_pred = (x - pred_x, y - pred_y)
    #   other_sigs = []

    #   for i in range(len(other_halos)):
    #     other_sigs.append(e_tang)
    #     sig_vect = [0, 0]

    #     for j in range(i+1):
    #       mag_dist_vect = np.sqrt((x-other_halos[j].x)**2 + (y-other_halos[j].y)**2)
    #       sig_vect[0] += (x-other_halos[j].x)/mag_dist_vect * other_sigs[j]
    #       sig_vect[1] += (y-other_halos[j].y)/mag_dist_vect * other_sigs[j]

    #     dot_prod = (sig_vect[0] * vect_to_pred[0] + sig_vect[1] * vect_to_pred[1])
    #     mag_sig_vect = np.sqrt(sig_vect[0]**2 + sig_vect[1]**2)
    #     mag_vect_to_pred = np.sqrt(vect_to_pred[0]**2 + vect_to_pred[1]**2)

    #     try:
    #       cos_phi = dot_prod / (mag_vect_to_pred * mag_sig_vect)
    #       e_tang = e_tang - mag_sig_vect*cos_phi
    #     except:
    #       e_tang = 0.0

    # return e_tang.mean()