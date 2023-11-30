import copy
import os
import random
from enum import Enum
import matplotlib.pyplot as plt
import time


class Orientation(Enum):
	Horizontal = 0,
	Vertical = 1,
	NotUsed = -1,


class META:
	grid_size = 20
	mutation_per = 0.3
	crossover_per = 0.6
	old_best_per = 0.1

	generations = 1000
	population_size = 500

	basic_score = 0
	penalty = 1


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.intersections: dict = {}

		self.orientation: Orientation = Orientation.NotUsed
		self.position: tuple = None
		self.end: tuple = None
		self.visited: bool = False
		self.valid = True

	def __lt__(self, other):
		return self.length > other.length

	def overlap(self, other) -> bool:
		par_1, par_2 = (1, 0) if self.orientation == Orientation.Vertical else (0, 1)

		if (self.position[par_1] <= other.position[par_1] <= self.end[par_1]
				or self.position[par_1] <= other.end[par_1] <= self.end[par_1]):
			if self.position[par_2] == other.position[par_2]:
				return True
		return False


def find_intersection_point(word1, word2):
	x1, y1 = word1.position
	x2, y2 = word1.end
	x3, y3 = word2.position
	x4, y4 = word2.end
	# For Lines with One Horizontal and One Vertical
	if (x1 == x2 and y3 == y4) or (y1 == y2 and x3 == x4):
		if x1 == x2:
			if min(x3, x4) <= x1 <= max(x3, x4) and min(y1, y2) <= y3 <= max(y1, y2):
				return x1, y3
		elif y1 == y2:
			if min(y3, y4) <= y1 <= max(y3, y4) and min(x1, x2) <= x3 <= max(x1, x2):
				return x3, y1
	return False


def dfs(word: Word):
	word.visited = True
	for word_ in word.intersections.values():
		if not word_.visited:
			dfs(word_)


def fitness(individual):
	crossword = copy.deepcopy(individual)
	score = META.basic_score
	for word in crossword:
		for word_ in crossword:
			if word_.string == word.string:
				continue

			if word_.orientation == word.orientation:
				par_1, par_2 = (1, 0) if word.orientation == Orientation.Vertical else (0, 1)

				if word.overlap(word_):
					score -= META.penalty

				elif word_.position[par_2] in (word.position[par_2] + 1, word.position[par_2] - 1):
					if (word.position[par_1] <= word_.position[par_1] < word.end[par_1]
							or word.position[par_1] < word_.end[par_1] <= word.end[par_1]):
						score -= META.penalty
				elif word_.position[par_2] == word.position[par_2]:
					if word_.end[par_1] == word.position[par_1] - 1 or word_.position[par_1] == word.end[par_1] + 1:
						score -= META.penalty
			else:
				intersect = find_intersection_point(word, word_)
				vert_w = word if word.orientation == Orientation.Vertical else word_
				horiz_w = word_ if word.orientation == Orientation.Vertical else word

				if intersect:
					if (vert_w.string[intersect[1] - vert_w.position[1]]
							!= horiz_w.string[intersect[0] - horiz_w.position[0]]):
						score -= META.penalty
					else:
						word.intersections[intersect] = word_
						word_.intersections[intersect] = word
				else:
					if vert_w.position[0] in (horiz_w.position[0] - 1, horiz_w.end[0] + 1):
						if vert_w.position[1] <= horiz_w.position[1] <= vert_w.end[1]:
							score -= META.penalty
					elif horiz_w.position[1] in (vert_w.position[1] - 1, vert_w.end[1] + 1):
						if horiz_w.position[0] <= vert_w.position[0] <= horiz_w.end[0]:
							score -= META.penalty

	for word in crossword:
		for word_ in crossword:
			if word_.string == word.string or word_.orientation != word.orientation:
				continue

			par_1, par_2 = (1, 0) if word.orientation == Orientation.Vertical else (0, 1)

			if word_.position[par_2] in (word.position[par_2] + 1, word.position[par_2] - 1):
				if word_.position[par_1] == word.end[par_1]:
					if not word.intersections.get(word.end) or not word_.intersections.get(word_.position):
						score -= META.penalty
				elif word.position[par_1] == word_.end[par_1]:
					if not word.intersections.get(word.position) or not word_.intersections.get(word_.end):
						score -= META.penalty

	# for i in range(len(crossword)):
	dfs(crossword[0])
	for word in crossword:
		if not word.visited or len(word.intersections) == 0:
			score -= META.penalty
		else:
			word.visited = False
	return score


def mutation(crossword):
	individual = copy.deepcopy(crossword)
	for i in range(3):
		rand_gen = random.choice(individual)
		random_position(rand_gen)
	return individual


def crossover(parent1, parent2):
	def two_points():
		# FIXME not working for 3 or less words.txt
		point1, point2 = sorted(random.sample(range(1, len(parent1) - 1), 2))
		offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
		offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
		return offspring1, offspring2

	def mixed():
		offspring1 = copy.deepcopy(parent1)
		offspring2 = copy.deepcopy(parent2)
		for index in range(len(parent1)):
			if random.choice([True, False]):
				offspring1[index], offspring2[index] = offspring2[index], offspring1[index]
		return offspring1, offspring2

	return mixed()


def best_old_individuals(population, fitness_arr):
	def get_max_indices(arr):
		n = round(len(population) * 0.3)
		sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k])
		max_indices = sorted_indices[-n:][::-1]
		return max_indices

	pop = copy.deepcopy(population)
	fit_arr = copy.deepcopy(fitness_arr)
	best_ind = [pop[i] for i in get_max_indices(fit_arr)]
	return best_ind


def next_generation(population, fitness_arr):
	best_old = best_old_individuals(population, fitness_arr)
	n = round(len(population) * META.old_best_per)

	new_population = copy.deepcopy(best_old[:n])

	for _ in range(round((len(population) * META.crossover_per) / 2)):
		offspring1, offspring2 = crossover(random.choice(best_old), random.choice(best_old))
		new_population.append(offspring1)
		new_population.append(offspring2)

	for _ in range(round(len(population) * META.mutation_per)):
		new_population.append(mutation(random.choice(population)))

	return new_population


def print_crossword(population):
	grid = []
	for _ in range(META.grid_size):
		grid.append(['-' for _ in range(META.grid_size)])
	for word in population:
		x, y = word.position
		for i in range(word.length):
			if word.orientation == Orientation.Horizontal:
				grid[y][x + i] = word.string[i]
			else:
				grid[y + i][x] = word.string[i]

	print("+" + "-" * (META.grid_size * 3 - 1) + "+")
	for row in grid:
		print("|" + "  ".join(row) + "|")
	print("+" + "-" * (META.grid_size * 3 - 1) + "+")


def best(population, fitness_arr):
	fit_arr = copy.deepcopy(fitness_arr)
	ind = fit_arr.index(max(fit_arr))
	return population[ind], fit_arr[ind]


def random_position(word):
	word.orientation = random.choice([Orientation.Horizontal, Orientation.Vertical])
	if word.orientation == Orientation.Horizontal:
		x = random.randint(0, META.grid_size - word.length)
		y = random.randint(0, META.grid_size - 1)
		x_ = x + word.length - 1
		y_ = y
	else:
		x = random.randint(0, META.grid_size - 1)
		y = random.randint(0, META.grid_size - word.length)
		x_ = x
		y_ = y + word.length - 1
	word.position = (x, y)
	word.end = (x_, y_)


def init_zero_population(w: list[Word]) -> list:
	zero_population = []
	for i in range(META.population_size):
		words = copy.deepcopy(w)
		for word in words:
			random_position(word)
		zero_population.append(words)
	return zero_population


def read_file(file_name):
	with open(file_name, "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


# def output(file_name, solved, crossword):
# 	with open(f"outputs/{file_name}.txt", 'w') as out:
# 		if solved:
# 			for word in crossword:
# 				out.write(f"{word.position[0]} {word.position[1]} {word.orientation.value[0]}")
# 		else:
# 			out.write("Not successful")


# def solve(file_name):
# 	words = read_file(file_name)
# 	print(f"Words: {len(words)}", end="  ")
# 	population = init_zero_population(words)
# 	x_data = []
# 	y_data = []
#
# 	loss_arr = []
#
# 	for generation in range(META.generations):
# 		fitness_arr = [fitness(individual) for individual in population]
# 		best_crossword, best_fitness = best(population, fitness_arr)
# 		population = next_generation(population, fitness_arr)
#
# 		if best_fitness == 0:
# 			plt.ylabel('Avr Loss')
# 			plt.xlabel('Generations')
# 			plt.scatter(x_data, y_data)
# 			plt.show()
# 			return True
#
# 		avr_loss = abs(sum(fitness_arr)) / len(fitness_arr)
# 		loss_arr.append(avr_loss)
#
# 		x_data.append(generation)
# 		y_data.append(avr_loss)
#
# 		if generation % 15 == 0:
# 			plt.ylabel('Avr Loss')
# 			plt.xlabel('Generations')
# 			plt.scatter(x_data, y_data)
# 			plt.show()
#
# 		if len(loss_arr) >= 40:
# 			if loss_arr[-1] >= 10:
# 				num = 40
# 				threshold = 0.3
# 			elif loss_arr[-1] >= 5:
# 				num = 40
# 				threshold = 0.3
# 			else:
# 				num = min(50, len(loss_arr))
# 				threshold = 0.2
# 			arr = [abs(loss_arr[i] - loss_arr[i - 1]) <= threshold for i in
# 				   range(len(loss_arr) - num, len(loss_arr))]
# 			if all(arr):
# 				# print("Restart")
# 				population = init_zero_population(words)
#
# 	plt.ylabel('Avr Loss')
# 	plt.xlabel('Generations')
# 	plt.scatter(x_data, y_data)
# 	plt.show()
# 	print("Bad", end="  ")


def create_new_population(best_crossword, w):
	population = []
	for _ in range(META.population_size):
		crossword = copy.deepcopy(best_crossword)
		word = copy.deepcopy(w)
		random_position(word)
		crossword.append(word)
		population.append(crossword)
	return population


def add_new_word(population, new_word):
	for crossword in population:
		word = copy.deepcopy(new_word)
		crossword.append(word)
		random_position(crossword[-1])
	return population


def test_solve(file_name):
	words = sorted(read_file(file_name))
	print("Words:", len(words), end=" ")

	while True:
		index = 2
		four_words = copy.deepcopy(words[:2])
		population = init_zero_population(four_words)

		generation = 0
		while generation < META.generations:
			fitness_arr = [fitness(individual) for individual in population]
			best_crossword, best_fitness = best(population, fitness_arr)

			population = next_generation(population, fitness_arr)
			if best_fitness == 0:
				if len(words) == len(population[0]):
					return len(words), abs(sum(fitness_arr)) / len(fitness_arr)
				generation = 0
				new_word = copy.deepcopy(words[index])
				population = create_new_population(population, new_word)

			if generation > 200:
				print("Again", end=" ")
				break
			generation += 1


if __name__ == "__main__":
	words_arr = []
	loss_arr = []
	time_arr = []

	tik = time.time()
	for i in os.listdir("inputs"):
		timer = time.time()
		words, loss = test_solve(f"inputs/{i}")
		timer = time.time() - timer

		loss_arr.append(loss)
		words_arr.append(words)
		time_arr.append(timer)

		sec = str(int(timer) % 60)
		if len(sec) == 1:
			sec = "0" + sec
		print(f"Time: {int(timer // 60)}:{sec}")
	print("Total time: ", time.time() - tik)

	plt.ylabel('Avr Loss')
	plt.xlabel('Words')
	plt.scatter(words_arr, loss_arr)
	plt.show()

	plt.ylabel('Time in sec')
	plt.xlabel('Words')
	plt.scatter(words_arr, time_arr)
	plt.show()
