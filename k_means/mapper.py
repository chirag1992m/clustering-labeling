import pickle, sys, numpy as np, math

data = pickle.load(sys.stdin.buffer)

centroids = np.array(data['centroids'])
points = np.array(data['points'])

updated_centroids = np.append(np.zeros_like(centroids), np.zeros((centroids.shape[0], 1)), axis=1)

def distance(a, b):
	return np.linalg.norm(a - b)

def get_nearest_centroid_id(p):
	min_distance = math.inf
	centroid_ID = -1
	for idx in range(centroids.shape[0]):
		if distance(centroids[idx], p) < min_distance:
			min_distance = distance(centroids[idx], p)
			centroid_ID = idx

	return idx

def extend_point(p):
	return np.append(p, 1)

for p in points.shape[0]:
	centroid_ID = get_nearest_centroid_id(points[p])
	new_p = extend_point(points[p])
	updated_centroids[centroid_ID] += new_p

for idx in range(updated_centroids.shape[0]):
	pickle.dump((idx, updated_centroids[idx]), sys.stdout.buffer)