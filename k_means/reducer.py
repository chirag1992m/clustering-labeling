import pickle, sys, numpy as np
from itertools import groupby
from operator import itemgetter

centroids = []

while True:
	try:
		centroids.append(pickle.load(sys.stdin.buffer))
	except EOFError:
		break

for center_ID, group in groupby(centroids, itemgetter(0)):
	points = np.array([point for _, point in group])
	new_center = np.sum(points, axis=0)
	new_center = get_mean(new_center)

	#Dump the new centers
	pickle.dump((center_ID, new_center))