from everything import *
from machine_learning import *
from data_processing import *
from sky import *
from metric import *

if __name__ == "__main__":
  # sky_range = [1, 101, 201]
  sky_range = None
  skies = objectify_data(test=False, sky_range=sky_range)
 

  for s in [1, 0.9, 0.8, 0.7, 0.6]:
  # for s in [1, 0.8]:
    training_file = "training_data_s=%i.csv" % int(s*10)
    print "Generating training data:%s" % training_file
    generate_halo_mag_data(training_file, sky_range=sky_range, scaling_factor=s)
    training_data = objectify_training_data(training_file)

    output_file = 'opt_scaling_factor_s=%i' % int(s*10)
    write_data(skies, output_file=output_file, method=Sky.better_subtraction, \
             opts={'to_file': True, 'training_data': training_data, 'scaling_factor': s})
    m = analyze(output_file, '../data/Training_halos.csv')
    print "Scaling factor: %f, metric: %f" % (s, m)
  # generate_halo_mag_data(sky_range=[1,101,201])

  # write_data(skies, output_file='random_number_of_halos.csv', method=Sky.non_binned_signal, opts={'to_file': True})

  # skies[0].plot()

  # skies[0].non_binned_signal()
  # skies[0].better_subtraction()
  # dist_between_halos()

  # analyze_ratio()
  # analyze_force_within_radius()