import json
import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
	words_fitness_files: dict = {}
	for i in range(10, 16):
		words_fitness_files[i] = []

	for js in sorted(os.listdir("logs")):
		with open(f"logs/{js}", "r") as file:
			data = json.load(file)
			words_fitness_files[int(data["Words"])].append(js)

	word_fit_max: dict = {}
	word_fit_avg: dict = {}
	word_time: dict = {}

	for i in range(10, 16):
		word_fit_max[i] = []
		word_fit_avg[i] = []
		word_time[i] = []

		for f in words_fitness_files[i]:
			with open(f"logs/{f}", "r") as file:
				data = json.load(file)
				word_fit_max[i].append(data["Max fitness"])
				word_fit_avg[i].append(data["Average final fitness"])
				word_time[i].append(data["Time (sec)"])

	a = [np.mean(stat) for stat in word_time.values()]
	x = [10, 11, 12, 13, 14, 15]

	plt.plot(x, a)
	plt.title("Time per words")
	plt.xlabel("Words")
	plt.ylabel("Time (sec.)")
	plt.show()
