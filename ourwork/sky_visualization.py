import numpy as np
import csv
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


  def plot(self, with_predictions=False):
    fig = figure()
    ax = fig.add_subplot(111, aspect='equal')

    for gal in sky.galaxies:
      e = Ellipse((gal.x, gal.y), gal.a, gal.b, angle=gal.theta, linewidth=2, fill=True)
      ax.add_artist(e)
      e.set_clip_box(ax.bbox)

    if (with_predictions):
      halos_a = self.gridded_signal()
      halos_b = self.max_likelihood()
      plt.plot(halos_a[0, 0], halos_a[1, 0], marker='o', linewidth=10, color='red')
      plt.plot(halos_b[0, 0], halos_b[1, 0], marker='o', linewidth=10, color='yellow')

    ax.set_xlim(0, 4200)
    ax.set_ylim(0, 4200)
    show()

  def gridded_signal(self):
    position_halo=np.zeros([2,3],float) #Set up the array in which I will
                                        #assign my estimated positions
    nhalo = 1

    nbin=15                 #Number of bins in my grid
    image_size=4200.0       #Overall size of my image
    binwidth=float(image_size)/float(nbin) # The resulting width of each grid section

    average_tan_force=np.zeros([nbin,nbin],float) #Set up the signal array
                                                  #in which Im going to find
                                                  #the maximum of.
    x = np.array([galaxy.x for galaxy in self.galaxies])
    y = np.array([galaxy.y for galaxy in self.galaxies])
    e1 = np.array([galaxy.e1 for galaxy in self.galaxies])
    e2 = np.array([galaxy.e2 for galaxy in self.galaxies])

    for i in xrange(nbin):
      for j in xrange(nbin):
        x0=i*binwidth+binwidth/2. #proposed x position of the halo
        y0=j*binwidth+binwidth/2. #proposed y position of the halo
    
        angle_wrt_halo=np.arctan((y-y0)/(x-x0)) #I find the angle each
                                            #galaxy is at with respects
                                            #to the centre of the halo.               
        tangential_force=-(e1*np.cos(2.0*angle_wrt_halo)\
                       +e2*np.sin(2.0*angle_wrt_halo))
                       #Find out what the tangential force
                       #(or signal) is for each galaxy with
                       #respects to the halo centre, (x0,y0)
        tangential_force_in_bin=tangential_force[(x >= i*binwidth) & \
                                             (x < (i+1)*binwidth) & \
                                             (y >= j*binwidth) & \
                                             (y < (j+1)*binwidth)]
                            #Find out which galaxies lie within the gridded box


        if len(tangential_force_in_bin) > 0:
          average_tan_force[i,j]=sum(tangential_force_in_bin)\
                /len(tangential_force_in_bin) #Find the average signal per galaxy
        else:
          average_tan_force[i,j]=0
            

    index=np.sort(average_tan_force,axis=None) #Sort the grid into the
                                               #highest value bin first,
                                               #which should be the centre
                                               #of one of the halos
    index=index[::-1] #Reverse the array so the largest is first
    for n in xrange(nhalo):
      position_halo[0,n]=np.where(average_tan_force\
                                      == index[n])[0][0]\
                                      *binwidth
                                      #For each halo in the sky find
                                      #the position and assign
      position_halo[1,n]=np.where(average_tan_force\
                                      == index[n])[1][0]\
                                      *binwidth
                                      #The three grid bins
                                      #with the highest signal should
                                      #contain the three halos.
    return position_halo


  def max_likelihood(self):
    halos=3 #Number of halos in the most complicated sky
    position_halo=np.zeros([2,halos],float) #The array in which
                #I will record the position
                #of each halo

    #   #Grid the sky up. Here I set the parameters of the grid.
    nbin=10 #Number of bins in my grid
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
            
            
    position_halo[0,0] = np.where(likelihood == \
            np.max(likelihood))[0][0]*binwidth
    position_halo[1,0] = np.where(likelihood == \
            np.max(likelihood))[1][0]*binwidth

    return position_halo


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
  for k in xrange(n_skies):
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

  # for sky in data:
    
  c = csv.writer(open("Gridded_Signal_benchmark_redo.csv", "wb")) #Now write the array to a csv file
  c.writerow([str('SkyId'),str('pred_x1'),str( 'pred_y1'),str( 'pred_x2'),str( 'pred_y2'),str( 'pred_x3'),str(' pred_y3')])
  for k in xrange(len(data)):
    halostr=['Sky'+str(k+1)] #Create a string that will write to the file
                    #and give the first element the sky_id
    pos_halo= data[k].gridded_signal()
    for n in xrange(3):
      halostr.append(pos_halo[0,n]) #Assign each of the
                                           #halo x and y positions to the string
      halostr.append(pos_halo[1,n])
    c.writerow(halostr) #Write the string to a csv
                        #file with the sky_id and the estimated positions


if __name__ == "__main__":
  plot_data()