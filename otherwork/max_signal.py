from pandas import *
import os
from matplotlib.pyplot import *
import numpy as np

# Load them files
def init():
	train_halos = read_csv('../data/Training_halos.csv')
	test_halo_counts = read_csv('../data/Test_haloCounts.csv')	
	test_skies = {}
	for s in os.listdir("../data/Test_Skies"):
		test_skies[s.split("_")[1].split(".")[0]] = read_csv("../data/Test_Skies/" + s)	
	train_skies = {}
	for s in os.listdir("../data/Train_Skies"):
		train_skies[s.split("_")[1].split(".")[0]] = read_csv("../data/Train_Skies/" + s)
	return train_halos, test_halo_counts,test_skies,train_skies

train_halos, test_halo_counts, test_skies, train_skies = init()

# Euclidean Distance
def d((x0,y0),(x1,y1)): return np.sqrt((x0-x1)**2+(y0-y1)**2)

# Tangential ellipticity (signal)
# Takes a sky (galaxies) and the x,y coord for one halo
def e_tang(galaxies,x,y):	
	phi = np.arctan((galaxies.y - y)/(galaxies.x - x))
	e_tang = -(galaxies.e1*np.cos(2*phi)+galaxies.e2*np.sin(2*phi))
	return e_tang.mean()

# Return halos on given sky
def halos(sky):
	xs = []
	ys = []
	th = train_halos[train_halos.SkyId==sky]
	xs.append(th.halo_x1.values[0])
	ys.append(th.halo_y1.values[0])
	if th.halo_x2.values[0]!=0:
		xs.append(th.halo_x2.values[0])
		ys.append(th.halo_y2.values[0])
	if th.halo_x3.values[0]!=0:
		xs.append(th.halo_x3.values[0])
		ys.append(th.halo_y3.values[0])
	true_coords = zip(xs,ys)
	return true_coords

# Ye plotting function
def p(skyid):
	tq = []
	sky = "Sky"+str(skyid)
	ts2 = train_skies[sky]
	max_e = 0.0
	coord = (0,0)
	# bin width = 100
	for y_r in range(0,4200,100):
		for x_r in range(0,4200,100):
			t = e_tang(ts2,x_r,y_r)	
			tq.append(t)
			if max_e<t:
				max_e = t
				coord = (x_r,y_r)
	tq = np.array(tq).reshape((42,42))
	# Mirror the image to have origin at bottom left of picture
	tq = tq[::-1]	
	print tq
	# Get first halo's (actual) coordinates
	x = train_halos[train_halos.SkyId==sky].halo_x1.values[0]
	y = train_halos[train_halos.SkyId==sky].halo_y1.values[0]
	
	# This is our guess
	x_p, y_p = coord
	
	# Neat title
	title(" ".join([sky,"True coords:","("+str(x)+","+str(y)+")","Pred coords:","("+str(x_p)+","+str(y_p)+")"]))
	imshow(tq,interpolation="nearest")
	xticks(range(0,42,5),range(0,4200,500))
	y_c = range(0,4200,500)
	y_c.reverse()
	yticks(range(1,42,5),y_c)
	true_coords = halos(sky)
	# Plot the true coordinates as red dots
	for x, y in true_coords:
		scatter(int(x/100),42-int(y/100),c="r")	# mirror y
	# Plot the 1 halo guess as a white dot
	scatter(int(x_p/100),42-int(y_p/100),c="w")	# mirror y

# For all skies
for s in range(16,17):
	p(s)	# plot them skies
	savefig('halos/'+str(s)+".png") #
	close()