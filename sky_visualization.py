import numpy as np
import matplotlib.pyplot as plt
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

def read_data():faf
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

  for k in [1]:#xrange(n_skies):
    p=k+1
    x,y,e1,e2=np.loadtxt('../data/Test_Skies/Test_Sky%i.csv' % p,\
             delimiter=',',unpack=True,usecols=(1,2,3,4),skiprows=1)
        #Read in the x,y,e1 and e2
        #positions of each galaxy in the list for sky number k:
    plt.plot(x, y, 'ro')
    plt.show()

if __name__ == "__main__":
  read_data()