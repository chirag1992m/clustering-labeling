import pickle, sys, scipy.sparse as sparse
from itertools import groupby
from operator import itemgetter

centroids = []

while True:
	try:
		centroids.append(pickle.load(sys.stdin.buffer))
	except EOFError:
		break

def get_mean(summed_points):
	size = summed_points.shape[0]
	summed_points = summed_points / summed_points[size-1]
	return summed_points[0:size-2]

for center_ID, group in groupby(centroids, itemgetter(0)):
	points = [point for _, point in group]
	if len(points) > 0:
		new_center = sparse.csr_matrix(shape=(1, points[0][0].shape[1]))

		total_points = 0
		for p in points:
			new_center += p[0]
			total_points += p[1]
		new_center /= total_points

	#Dump the new centers
	pickle.dump((center_ID, new_center))