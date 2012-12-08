from everything import *
from machine_learning import *
# from data_processing import *
from sky import *
from metric import *

if __name__ == "__main__":
  # sky_range = [1, 101, 201]
  sky_range = None 
  skies, test = objectify_data(sky_range=sky_range)
  

  output_file = 'universal_predictions.csv'
  # universe = Universe(skies, test=test)
  # universe.overguess_positions_of_halos()
  universe.write_predictions_to_file(output_file)

  m = analyze(output_file, '../data/Training_halos.csv')
  print m