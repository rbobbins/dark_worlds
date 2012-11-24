import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse 
import numpy as np
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
    self.determine_axes()

  def determine_axes(self):
    theta = np.arctan(self.e2/self.e1) / 2
    self.theta = theta * 180 / np.pi

    E = self.e1 / np.cos(2 * theta)
    self.a = 50/(1-E)
    self.b = 50/(1 + E)


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

  def plot_galaxies(self, ax):
    for gal in self.galaxies:
      e = Ellipse((gal.x, gal.y), gal.a, gal.b, angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)

  def non_binned_signal(self, to_file=False, nhalos=0):
    if nhalos == 0:
      nhalos = random.choice([1,2,3])
    nhalos = 2

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
    
    halos, tqs = self.non_binned_signal()
    halo1, halo2, halo3 = halos
    tq1, tq2, tq3 = tqs

    fig = figure(figsize=(11,11)) 
    for tq, subplotid in [(tq1, 221), (tq2, 222), (tq3, 223)]:
      
      #plot map of signal
      if tq != None: 
        ax = fig.add_subplot(subplotid, aspect='equal')
        plt.title(self.skyid)

        tq = np.array(tq).reshape((len(x_r_range),len(y_r_range)))
        plt.contourf(x_rs, y_rs, tq, 20)

        self.plot_galaxies(ax)

        for i, color in enumerate(['black', 'blue', 'pink']):
          if self.actual[i] != None:
            plt.plot(self.actual[i].x, self.actual[i].y, marker='*', markersize=20, color=color)

        if halo1: plt.plot(halo1.x, halo1.y, marker='o', markersize=10, color='black')
        if halo2: plt.plot(halo2.x, halo2.y, marker='o', markersize=10, color='blue')
        if halo3: plt.plot(halo3.x, halo3.y, marker='o', markersize=10, color='pink')
     
    show()
    return halos


  def e_tang(self, pred_x, pred_y, other_halos=[]):
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])
    
    theta = np.arctan((y - pred_y)/(x - pred_x))
    e_tang = -(e1 * np.cos(2 * theta) + e2 * np.sin(2 * theta))

    if len(other_halos) == 1:
      halo = other_halos[0]
      vect_to_halo = (x - halo.x, y - halo.y)
      vect_to_pred = (x - pred_x, y - pred_y)

      #find the projection of the vector to halo1 on the vector to 
      #(potential) halo 2.
      try:
        dot_prod = (vect_to_halo[0] * vect_to_pred[0] + vect_to_halo[1] * vect_to_pred[1])
        mag_vect_to_halo = np.sqrt(vect_to_halo[0]**2 + vect_to_halo[1]**2)
        mag_vect_to_pred = np.sqrt(vect_to_pred[0]**2 + vect_to_pred[1]**2)

        cos_phi = dot_prod / (mag_vect_to_pred * mag_vect_to_halo)
        e_tang = e_tang * (1 - cos_phi)
      except:
        return 0
    
    elif len(other_halos) == 2:
      halo1 = other_halos[0]
      halo2 = other_halos[1]
      # vect_to_halos = (x - (halo1.x + halo2.x), y - (halo1.y + halo2.y))
      # vect_to_halo1 = (x - halo1.x, y - halo1.y)
      vect_to_halo2 = (x - halo2.x, y - halo2.y)
      # vect_to_halos = vect_to_halo1
      vect_to_halos = vect_to_halo2
      # vect_to_halos = (vect_to_halo1[0] + vect_to_halo2[0], vect_to_halo1[1] + vect_to_halo2[1])
      vect_to_pred = (x - pred_x, y - pred_y)

      try:
        dot_prod = (vect_to_halos[0] * vect_to_pred[0] + vect_to_halos[1] * vect_to_pred[1])
        mag_vect_to_halos = np.sqrt(vect_to_halos[0]**2 + vect_to_halos[1]**2)
        mag_vect_to_pred = np.sqrt(vect_to_pred[0]**2 + vect_to_pred[1]**2)
        cos_phi = dot_prod / (mag_vect_to_pred*mag_vect_to_halos)
        e_tang = e_tang * (1 - cos_phi)
      except:
        return 0

    # for halo in other_halos:
    #   vect_to_halo = (x - halo.x, y - halo.y)
    #   vect_to_pred = (x - pred_x, y - pred_y)

    #   try:
    #     dot_prod = (vect_to_halo[0] * vect_to_pred[0]) + (vect_to_halo[1] * vect_to_pred[1])
    #     mag_vect_to_h1 = np.sqrt(vect_to_halo[0]**2 + vect_to_halo[1]**2)
    #     mag_vect_to_pred = np.sqrt(vect_to_pred[0]**2 + vect_to_pred[1]**2)
    #     cos_phi = dot_prod / (mag_vect_to_pred*mag_vect_to_h1)
    #     e_tang = e_tang * (1 - cos_phi)
    #   except:
    #     return 0

    return e_tang.mean()