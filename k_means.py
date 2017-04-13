#Load data
import os

print("Loading data...")
os.system("python -m clustering-labeling.load_data")
print("Data Loaded!")

for f in os.listdir("clustering-labeling/20_newsgroup/"):
	if f.endswith(".out"):
		os.rename("clustering-labeling/20_newsgroup/" + f, "clustering-labeling/k_means/" + f)

os.system("python -m clustering-labeling.coordinator --mapper_path=clustering-labeling/k_means/mapper.py --reducer_path=clustering-labeling/k_means/reducer.py --job_path=clustering-labeling/20_newsgroup --num_reducers=5 --timeout=100000")