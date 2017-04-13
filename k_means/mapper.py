import pickle, sys, math, os
import scipy.sparse as sparse, scipy.sparse.linalg as linalg
import time

start = time.time()

def load_clusters():
	clusters = []
	path = "clustering-labeling/k_means/"
	for f in os.listdir(path):
		if f.endswith(".out"):
			with open(path + f, "rb") as c:
				while True:
					try:
						clusters.append(pickle.load(c))
					except EOFError:
						break
	clusters.sort(key=lambda x:x[0])
	return [x[1] for x in clusters]

centroids = load_clusters()
points = pickle.load(sys.stdin.buffer)

centroid_points = [[sparse.csr_matrix((1, centroids[0].shape[1])), 0]] * len(centroids)

def distance(a, b):
	return linalg.norm(a - b)

def get_nearest_centroid_id(p):
	min_distance = math.inf
	centroid_ID = -1
	for idx, centroid in enumerate(centroids):
		if distance(centroid, p) < min_distance:
			min_distance = distance(centroid, p)
			centroid_ID = idx

	return idx

for p in range(points.shape[0]):
	centroid_ID = get_nearest_centroid_id(points[p])
	centroid_points[centroid_ID][0] += points[p]
	centroid_points[centroid_ID][1] += 1

print("Finished calculation! {}".format(time.time() - start), file=sys.stderr)

for idx in range(len(centroid_points)):
	pickle.dump((idx, centroid_points[idx]), sys.stdout.buffer)

print("Script End! {}".format(time.time() - start), file=sys.stderr)