import csv
import numpy as np
from everything import *

class DataPoint:
  def __init__(self, x, y):
    self.x = x #should be a np.array()
    self.y = y #should be an integer


def objectify_training_data(fname):
  read_file = csv.reader(open(fname, 'rb'))
  training_data = []
  
  #group data as training/cross_validation
  for row in read_file:
    if row[0] == 'n_actual_halos': continue  #ignore header row
    
    y = float(row[0])
    x1, x2, x3, x4, x5 = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
    ratios = [x2/x1, x3/x2, x4/x3, x5/x4, \
              x3/x1, x4/x2, x5/x3, \
              x4/x1, x5/x2, \
              x5/x1]
    X = np.array(ratios)
    training_data.append(DataPoint(X, y))
  return training_data
