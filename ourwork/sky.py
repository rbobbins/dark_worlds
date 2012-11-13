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
  def add_galaxy(self, galaxy):
    self.galaxies.append(galaxy)


  def plot(self, with_predictions=False):
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')

    for gal in self.galaxies:
      e = Ellipse((gal.x, gal.y), gal.a, gal.b, angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)

    for i in xrange(3):
      if self.actual[i] != None:
        plt.plot(self.actual[i][0], self.actual[i][1], marker='*', markersize=20, color='black')

    if (with_predictions):
      (bin_x0, bin_y0, average_tan_force) = self.gridded_signal_map()
      plt.contourf(bin_x0, bin_y0, average_tan_force)
      plt.colorbar()
      halos_a = self.gridded_signal()
      plt.plot(halos_a[0][0], halos_a[0][1], marker='o', linewidth=10, color='red')

      # halos_b = self.max_likelihood()
      # plt.plot(halos_b[0, 0], halos_b[1, 0], marker='o', linewidth=10, color='yellow')

    ax.set_xlim(0, 4200)
    ax.set_ylim(0, 4200)
    show()

  def gridded_signal_map(self, nbin=25, radius=560, radius_weight=0):
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
    
        angle_wrt_halo=np.arctan((y-bin_y0[i,j])/(x-bin_x0[i,j])) #I find the angle each
                                            #galaxy is at with respects
                                            #to the centre of the halo.               
        tangential_force = -(e1*np.cos(2.0*angle_wrt_halo)\
                       +e2*np.sin(2.0*angle_wrt_halo))
                       #Find out what the tangential force
                       #(or signal) is for each galaxy with
                       #respects to the halo centre, (bin_x0[i,j],bin_y0[i,j])
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

  def gridded_signal(self, nbin=25, radius=560, radius_weight=0):
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

  def max_likelihood(self, nbin=15):
    halos=3 #Number of halos in the most complicated sky
    # position_halo=np.zeros([2,halos],float) #The array in which
    position_halo = [(0,0), (0, 0), (0, 0)]
                #I will record the position
                #of each halo

    #   #Grid the sky up. Here I set the parameters of the grid.
    image_size=4200.0 #Overall size of my image
    binwidth=image_size/float(nbin) # The resulting width of each grid section
    likelihood=np.zeros([nbin,nbin],float) #The array in which I am going
                   #to store the likelihood that
                   #a halo could be at that
                   #grid point in the sky.

    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])
    
    for i in xrange(nbin): # I iterate over each x0
      for j in xrange(nbin): #and y0 points of the grid
            
        x0=i*binwidth #I set the proposed x position of the halo
        y0=j*binwidth #I set the proposed y position of the halo
            
        r_from_halo=np.sqrt((x-x0)**2+(y-y0)**2)
        # I find the distance each galaxy is from
        #the proposed halo position
        angle_wrt_centre=np.arctan((y-y0)/(x-x0))
        #I find the angle each galaxy is at with
        #respects to the centre of the halo.               
        force=1./r_from_halo
        #Assuming that the force dark matter has
        #on galaxies is 1/r i find the distortive
        #force the dark matter has on each galaxy
        
        e1_force=-1.*force*np.cos(2.*angle_wrt_centre)
        # work out what this force is in terms of e1
        e2_force=-1.*force*np.sin(2.*angle_wrt_centre) # and e2
            
        chisq=np.sum(((e1_force-e1)**2+(e2_force-e2)**2))
        # I then compare this hypotehtical e1 and e2
        #to the actual data in the sky and find the chisquare fit
        likelihood[i,j]=np.exp(-(chisq/2.))
        # I then find the likelihood that a halo is at position x0,y0
            
            
    x = (np.where(likelihood == np.max(likelihood))[0][0]*binwidth).tolist()
    y = (np.where(likelihood == np.max(likelihood))[1][0]*binwidth).tolist()

    position_halo[0] = (x, y)
    return position_halo

  def combined_approach(self):
    result1 = self.gridded_signal()
    result2 = self.max_likelihood()
    position_halos = [(0,0), (0,0), (0, 0)]

    # for i in [0, 1, 2]:
    x1, y1 = result1[0]
    x2, y2 = result2[0]

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    if distance < 200:
      # x = np.average([x1, x2])
      # y = np.average([y1, y2])
      position_halos[0] = (x1, y1)
    else:
      position_halos[0] = (x1, y1)
      position_halos[1] = (x2, y2)
    return position_halos