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
	crossover_per = 0.6 / 2
	old_best_per = 0.1

	generations = 80
	population_size = 500

	basic_score = 0
	penalty = 1

	output_path = "outputs"
	input_path = "inputs"


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.intersections: dict = {}

		self.orientation: Orientation = Orientation.NotUsed
		self.position: tuple = None
		self.end: tuple = None
		self.visited: bool = False

	def __lt__(self, other) -> bool:
		return self.length > other.length

	def overlap(self, other) -> bool:
		par_1, par_2 = (1, 0) if self.orientation == Orientation.Vertical else (0, 1)

		if (self.position[par_1] <= other.position[par_1] <= self.end[par_1]
				or self.position[par_1] <= other.end[par_1] <= self.end[par_1]):
			if self.position[par_2] == other.position[par_2]:
				return True
		return False

	def are_intersect(self, other) -> bool | tuple[int]:
		x1, y1 = self.position
		x2, y2 = self.end
		x3, y3 = other.position
		x4, y4 = other.end
		# For Lines with One Horizontal and One Vertical
		if (x1 == x2 and y3 == y4) or (y1 == y2 and x3 == x4):
			if x1 == x2:
				if min(x3, x4) <= x1 <= max(x3, x4) and min(y1, y2) <= y3 <= max(y1, y2):
					return x1, y3
			elif y1 == y2:
				if min(y3, y4) <= y1 <= max(y3, y4) and min(x1, x2) <= x3 <= max(x1, x2):
					return x3, y1
		return False

	def random_position(self):
		self.orientation = random.choice([Orientation.Horizontal, Orientation.Vertical])
		if self.orientation == Orientation.Horizontal:
			x = random.randint(0, META.grid_size - self.length)
			y = random.randint(0, META.grid_size - 1)
			x_ = x + self.length - 1
			y_ = y
		else:
			x = random.randint(0, META.grid_size - 1)
			y = random.randint(0, META.grid_size - self.length)
			x_ = x
			y_ = y + self.length - 1
		self.position = (x, y)
		self.end = (x_, y_)


class Crossword:
	def __init__(self, words):
		self.words: list[Word] = words
		self.words_num = len(self.words)

	def dfs(self, word: Word):
		word.visited = True
		for word_ in word.intersections.values():
			if not word_.visited:
				self.dfs(word_)

	def fitness(self) -> int:
		crossword = copy.deepcopy(self.words)
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
					intersect = word.are_intersect(word_)
					vert_w = word if word.orientation == Orientation.Vertical else word_
					horiz_w = word_ if word.orientation == Orientation.Vertical else word

					if intersect:
						if (vert_w.string[intersect[1] - vert_w.position[1]]
								!= horiz_w.string[intersect[0] - horiz_w.position[0]]):
							score -= META.penalty
						else:
							word.intersections[intersect] = word_
					else:
						if vert_w.position[0] in (horiz_w.position[0] - 1, horiz_w.end[0] + 1):
							if vert_w.position[1] <= horiz_w.position[1] <= vert_w.end[1]:
								score -= META.penalty
						elif horiz_w.position[1] in (vert_w.position[1] - 1, vert_w.end[1] + 1):
							if horiz_w.position[0] <= vert_w.position[0] <= horiz_w.end[0]:
								score -= META.penalty

		# Check the case of parallel neighbour words, for first and last symbols in the words
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

		for i in range(self.words_num):
			self.dfs(crossword[i])
			for word in crossword:
				if not word.visited or len(word.intersections) == 0:
					score -= META.penalty
				else:
					word.visited = False
		return score

	def mutation(self):
		individual = copy.deepcopy(self.words)
		rand_gen = random.choice(individual)
		rand_gen.random_position()
		return Crossword(individual)

	def crossover(self, parent2) -> tuple[Word]:
		def two_points():
			point1, point2 = sorted(random.sample(range(0, len(self.words) - 1), 2))
			offspring1 = Crossword(self.words[:point1] + parent2.words[point1:point2] + self.words[point2:])
			offspring2 = Crossword(parent2.words[:point1] + self.words[point1:point2] + parent2.words[point2:])
			return offspring1, offspring2

		def one_point():
			point = random.randint(0, self.words_num - 1)
			offspring1 = Crossword(self.words[:point] + parent2.words[point:])
			offspring2 = Crossword(parent2.words[:point] + self.words[point:])
			return offspring1, offspring2

		return two_points() if self.words_num > 2 else one_point()

	def print(self):
		grid = []
		for _ in range(META.grid_size):
			grid.append(['-' for _ in range(META.grid_size)])
		for word in self.words:
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

	def insert_word(self, word: Word):
		self.words.append(word)
		self.words_num += 1

	def output_solution(self, file_name):
		with open(f"{META.output_path}/{file_name}", "w") as file:
			for word in self.words:
				file.write(f"{word.position[0]} {word.position[1]} {word.orientation.value}\n")


def best_old_individuals(population: list[Crossword], fitness_arr: list[int]) -> list[Crossword]:
	def get_max_indices(arr):
		n = round(len(population) * 0.3)
		sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k])
		max_indices = sorted_indices[-n:][::-1]
		return max_indices

	pop = copy.deepcopy(population)
	fit_arr = copy.deepcopy(fitness_arr)
	best_ind = [pop[i] for i in get_max_indices(fit_arr)]
	return best_ind


def evolve(population: list[Crossword], fitness_arr: list[int]) -> list[Crossword]:
	best_old = best_old_individuals(population, fitness_arr)
	n = round(len(population) * META.old_best_per)

	new_population = copy.deepcopy(best_old[:n])

	for _ in range(round((len(population) * META.crossover_per))):
		offspring1, offspring2 = random.choice(best_old).crossover(random.choice(best_old))
		new_population.append(offspring1)
		new_population.append(offspring2)

	for _ in range(round(len(population) * META.mutation_per)):
		kek = random.choice(population)
		new_population.append(kek.mutation())

	for cr in new_population:
		if not isinstance(cr, Crossword):
			print(0)

	return new_population


def best_fit_crossword(population: list[Crossword], fitness_arr: list[int]) -> tuple[Crossword, int]:
	fit_arr = copy.deepcopy(fitness_arr)
	ind = fit_arr.index(max(fit_arr))
	return population[ind], fit_arr[ind]


def init_zero_population(input_words: Crossword) -> list[Crossword]:
	zero_population = []
	for _ in range(META.population_size):
		words = copy.deepcopy(input_words)
		for word in words:
			word.random_position()
		zero_population.append(Crossword(words=words))
	return zero_population


def create_new_population(best_crossword: Crossword, new_word: Word):
	population = []
	for _ in range(META.population_size):
		crossword = copy.deepcopy(best_crossword)
		word = copy.deepcopy(new_word)
		word.random_position()
		crossword.insert_word(word)
		population.append(crossword)
	return population


def read_file(file_name) -> Crossword:
	with open(f"{META.input_path}/{file_name}", "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


def solve(file_name):
	words = sorted(read_file(file_name))
	print("Words:", len(words), end=" ")

	while True:
		index = 2
		initial_words_set = copy.deepcopy(words[:index])
		population = init_zero_population(initial_words_set)

		for generation in range(META.generations):
			fitness_arr = [individual.fitness() for individual in population]
			best_crossword, best_fitness = best_fit_crossword(population, fitness_arr)
			population = evolve(population, fitness_arr)
			if best_fitness == 0:
				if len(words) == population[0].words_num:
					best_crossword.output_solution(file_name)
					return len(words), abs(sum(fitness_arr)) / len(fitness_arr)
				new_word = copy.deepcopy(words[index])
				population = create_new_population(best_crossword, new_word)
				index += 1


if __name__ == "__main__":
	words_arr = []
	loss_arr = []
	time_arr = []

	total_t = time.time()
	for file in os.listdir(META.input_path):
		timer = time.time()
		words, loss = solve(file)
		timer = time.time() - timer

		loss_arr.append(loss)
		words_arr.append(words)
		time_arr.append(timer)

		sec = str(int(timer) % 60)
		if len(sec) == 1:
			sec = "0" + sec
		print(f"Time: {int(timer // 60)}:{sec}")
	print("Total time: ", time.time() - total_t)

	plt.ylabel('Avr Loss')
	plt.xlabel('Words')
	plt.scatter(words_arr, loss_arr)
	plt.show()

	plt.ylabel('Time in sec')
	plt.xlabel('Words')
	plt.scatter(words_arr, time_arr)
	plt.show()
