import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse 
import numpy as np
from pylab import figure, show, rand

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
  def __init__(self, halo1=None, halo2=None, halo3=None):
    self.galaxies = []
    self.predictions = [(0, 0), (0, 0), (0, 0)]
    self.actual = [halo1, halo2, halo3]
  
  def n_halos(self):
    return sum([1 for h in self.actual if h != None])
  
  def add_galaxy(self, galaxy):
    self.galaxies.append(galaxy)

  def plot_galaxies(self, ax):
    # fig = figure() 
    # ax = fig.add_subplot(111, aspect='equal')
    for gal in self.galaxies:
      e = Ellipse((gal.x, gal.y), gal.a, gal.b, angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)

  def non_binned_signal(self, to_file=False):
    max_e = 0.0
    tq = []
    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    #predicts the first halo
    # x1, y1 = None, None
    if self.actual[0] != None:
      for y_r in y_r_range:
        for x_r in x_r_range:
          t = self.e_tang(x_r,y_r) 
          tq.append(t)
          if max_e<t:
            max_e = t
            p1 = x_r, y_r

    # x2, y2 = None, None
    p2, tq2 = None, None
    if self.actual[1] != None:
      # print "going through point 2"
      max_e_halo2 = 0.0
      tq2 = []
      for iy, y_r in enumerate(y_r_range):
        for ix, x_r in enumerate(x_r_range):
          if p1[0] == x_r and p1[1] == y_r: tq2.append(0); continue

          t = self.e_tang(x_r, y_r, [p1])
          tq2.append(t)
          
          if max_e_halo2 < t:
            max_e_halo2 = t
            p2 = x_r, y_r

    p3, tq3 = None, None
    if self.actual[2] != None:
      max_e_halo3 = 0.0
      tq3 = []
      for iy, y_r in enumerate(y_r_range):
        for ix, x_r in enumerate(x_r_range):
          if p1[0] == x_r and p1[1] == y_r: tq3.append(0); continue
          if p2[0] == x_r and p2[1] == y_r: tq3.append(0); continue

          i = ix * len(y_r_range) + iy

          t = self.e_tang(x_r, y_r, [p1, p2])
          tq3.append(t)

          if max_e_halo3 < t:
            max_e_halo3 = t
            p3 = x_r, y_r

    if to_file == True:
      return p1, p2, p3
    else:
      return [p1, p2, p3], tq, tq2, tq3

  def plot(self):
    fig = figure() 
    ax = fig.add_subplot(111, aspect='equal')

    self.plot_galaxies(ax)
    for i, color in enumerate(['black', 'blue', 'pink']):
      if self.actual[i] != None:
        plt.plot(self.actual[i][0], self.actual[i][1], marker='*', markersize=20, color=color)

    x_r_range = range(0,4200,70)
    y_r_range = range(0,4200,70)
    
    points, tq1, tq2, tq3 = self.non_binned_signal()
    p1, p2, p3 = points
 
    #plot map of signal
    if tq1 != None: tq1 = np.array(tq1).reshape((len(x_r_range),len(y_r_range)))
    if tq2 != None: tq2 = np.array(tq2).reshape((len(x_r_range), len(y_r_range)))
    if tq3 != None: tq3 = np.array(tq3).reshape((len(x_r_range), len(y_r_range)))
    tq2 = (tq1 + tq2)
    x_rs, y_rs = np.meshgrid(x_r_range, y_r_range)
    plt.contourf(x_rs, y_rs, tq3, 20)
    
    #plot predicted positions
    if p1: plt.plot(p1[0], p1[1], marker='o', markersize=10, color='black')
    if p2: plt.plot(p2[0], p2[1], marker='o', markersize=10, color='blue')
    if p3: plt.plot(p3[0], p3[1], marker='o', markersize=10, color='pink')
   
    show()

    return points

  def e_tang(self, pred_x, pred_y, actual_halos=[]):
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])
    
    theta = np.arctan((y - pred_y)/(x - pred_x))
    e_tang = -(e1 * np.cos(2 * theta) + e2 * np.sin(2 * theta))

    if len(actual_halos) == 1:
      x_h, y_h = actual_halos[0]
      vect_to_halo = (x - x_h, y - y_h)
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
    
    elif len(actual_halos) == 2:
      x_h1, y_h1 = actual_halos[0]
      x_h2, y_h2 = actual_halos[1]
      # vect_to_halos = (x - (x_h1 + x_h2), y - (y_h1 + y_h2))
      # vect_to_halo1 = (x - x_h1, y - y_h1)
      vect_to_halo2 = (x - x_h2, y - y_h2)
      # vect_to_halos = (x - x_h1 )
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

    # for halo in actual_halos:
    #   x_h, y_h = halo
    #   vect_to_halo = (x - x_h, y - y_h)
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

  def gridded_signal_map(self, nbin=42, radius=200, radius_weight=0.0):
    image_size=4200.0       #Overall size of my image
    binwidth=float(image_size)/float(nbin) # The resulting width of each grid section

    bin_x0 = np.zeros([nbin,nbin],float) # Centers of each bin, x-coord
    bin_y0 = np.zeros([nbin,nbin],float) # Centers of each bin, y-coord

    average_bin_tan_force=np.zeros([nbin,nbin],float)
    average_radius_tan_force=np.zeros([nbin,nbin],float)
    average_tan_force=np.zeros([nbin,nbin],float) #Set up the signal array
                                                  #in which Im going to find
                                                  #the maximum of.
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])

    for i in xrange(nbin):
      for j in xrange(nbin):
        bin_x0[i,j]=i*binwidth+binwidth/2. #proposed x position of the halo
        bin_y0[i,j]=j*binwidth+binwidth/2. #proposed y position of the halo
    
        tangential_force = self.e_tang(bin_x0, bin_y0)

        tangential_force_in_bin = tangential_force[ (x >= i*binwidth) & \
                                                    (x < (i+1)*binwidth) & \
                                                    (y >= j*binwidth) & \
                                                    (y < (j+1)*binwidth)]
                            #Find out which galaxies lie within the gridded box
        tangential_force_in_radius = tangential_force[(np.abs(x-bin_x0[i,j]) >= binwidth/2) & \
                                                      (np.abs(x-bin_x0[i,j]) < radius) & \
                                                      (np.abs(y-bin_y0[i,j]) >= binwidth/2) & \
                                                      (np.abs(y-bin_y0[i,j]) < radius)]

        if len(tangential_force_in_bin) > 0:
          average_bin_tan_force[i,j] = sum(tangential_force_in_bin)/len(tangential_force_in_bin)
        else:
          average_bin_tan_force[i,j] = 0

        if len(tangential_force_in_radius) > 0:
          average_radius_tan_force[i,j] = sum(tangential_force_in_radius)/len(tangential_force_in_radius)
        else:
          average_radius_tan_force[i,j] = 0;

    average_tan_force = radius_weight*average_radius_tan_force + average_bin_tan_force
    return (bin_x0, bin_y0, average_tan_force)

  def gridded_signal(self, nbin=42, radius=300, radius_weight=0.5):
    position_halo = [(0,0), (0,0), (0,0)]
    nhalo = 3

    (bin_x0, bin_y0, average_tan_force) = self.gridded_signal_map(nbin, radius, radius_weight)

    index=np.sort(average_tan_force,axis=None) #Sort the grid into the
                                               #highest value bin first,
                                               #which should be the centre
                                               #of one of the halos
    index=index[::-1] #Reverse the array so the largest is first
    std = np.std(index)

    for n in xrange(nhalo):
      if (index[n] > index[0]-std):
        x = np.where(average_tan_force == index[n])[0][0] * (bin_x0[1] - bin_x0[0])[0]
        y = np.where(average_tan_force == index[n])[1][0] * (bin_x0[1] - bin_x0[0])[0]
        position_halo[n] = (x, y)

    return position_halo