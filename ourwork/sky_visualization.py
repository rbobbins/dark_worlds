import numpy as np
# import math
import matplotlib.pyplot as plt
from pylab import figure, show, rand
from matplotlib.patches import Ellipse 

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
  def __init__(self):
    self.galaxies = []

  def add_galaxy(self, galaxy):
    self.galaxies.append(galaxy)


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

def objectify_data():
  n_skies=file_len('../data/Test_haloCounts.csv')-1 #The number of skies in total
  halos=3 #Number of halos in the most complicated sky
  position_halo=np.zeros([n_skies,2,halos],float) #The array in which
              #I will record the position
              #of each halo

  nhalo = np.zeros([n_skies],float)
  col=np.zeros([1],int)+1
  nhalo=np.loadtxt('../data/Test_haloCounts.csv',\
       usecols=(1,),skiprows=1,delimiter=',')

            #Read in num_halos
  res = []
  for k in [1]:#xrange(n_skies):
    sky = Sky()
    p=k+1
    x,y,e1,e2=np.loadtxt('../data/Test_Skies/Test_Sky%i.csv' % p,\
             delimiter=',',unpack=True,usecols=(1,2,3,4),skiprows=1)
        #Read in the x,y,e1 and e2
        #positions of each galaxy in the list for sky number k:
    for i in range(len(x) - 1):
      sky.add_galaxy(Galaxy(x[i], y[i], e1[i], e2[i]))
    res.append(sky)
  return res

def plot_data():
  data = objectify_data()

  for sky in data:
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')
    for gal in sky.galaxies:
      e = Ellipse((gal.x, gal.y), gal.a, gal.b, angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)
    ax.set_xlim(0, 4200)
    ax.set_ylim(0, 4200)
    show()


if __name__ == "__main__":
  plot_data()