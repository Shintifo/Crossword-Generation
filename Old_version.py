import copy
import random
import numpy as np
from enum import Enum


class Orientation(Enum):
	Horizontal = 0,
	Vertical = 1,
	NotUsed = -1


class META:
	grid_size = 20
	mutation_per = 0.05
	crossover_coef = 0.8
	old_best_coef = 1 - mutation_per - crossover_coef
	generations = 500
	gen_size = 1000

	basic_score = 3000

	penalty_touch = 10
	bad_intersection = 10
	penalty_nothing = 10
	penalty_overlap = 10


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.neighbours = {}  # Key - (char, num)  Value - Word
		self.orientation = Orientation.NotUsed
		self.position: tuple = None
		self.end: tuple = None

		self.visited = False
		self.intersections = []

	def __lt__(self, other):
		if self.position[0] < other.position[0]:
			return True
		elif self.position[0] == other.position[0] and self.position[1] < other.position[1]:
			return True
		else:
			return False

	def overlap(self, other) -> bool:
		par_1 = 1 * (self.orientation == Orientation.Vertical)
		par_2 = 1 * (par_1 == 0)

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
	for word_ in word.intersections:
		if not word_.visited:
			dfs(word_)


def fitness(crossword):
	# print_crossword(crossword)
	penalty = META.basic_score
	for word in crossword:
		for word_ in crossword:
			if word_.string == word.string:
				continue

			if word_.orientation == word.orientation:
				par_1 = 1 * (word.orientation == Orientation.Vertical)
				par_2 = 1 * (par_1 == 0)

				if word.overlap(word_):  # Overlapping
					penalty -= META.penalty_overlap

				elif word_.position[par_2] in (word.position[par_2] + 1, word.position[par_2] - 1):
					if (word.position[par_1] <= word_.position[par_1] <= word.end[par_1]
							or word.position[par_1] <= word_.end[par_1] <= word.end[par_1]):
						# TODO too close, but intersect with 3 word, so it's good
						penalty -= META.penalty_touch

			else:
				intersect = find_intersection_point(word, word_)
				vert_w = word if word.orientation == Orientation.Vertical else word_
				horiz_w = word_ if word.orientation == Orientation.Vertical else word

				if intersect:
					if (vert_w.string[intersect[1] - vert_w.position[1]]
							!= horiz_w.string[intersect[0] - horiz_w.position[0]]):
						penalty -= META.bad_intersection
					vert_w.intersections.append(horiz_w)
				else:
					if vert_w.position[0] in (horiz_w.position[0] - 1, horiz_w.end[0] + 1):
						if vert_w.position[1] <= horiz_w.position[1] <= vert_w.end[1]:
							penalty -= META.penalty_touch
					elif horiz_w.position[1] in (vert_w.position[1] - 1, vert_w.end[1] + 1):
						if horiz_w.position[0] <= vert_w.position[0] <= horiz_w.end[0]:
							penalty -= META.penalty_touch
	# print_crossword(crossword)
	dfs(crossword[0])
	for word in crossword:
		if word.visited:
			word.visited = False
		else:
			penalty -= META.penalty_nothing
	return max(0, penalty)


def mutation(individual):
	rand_index = random.randint(0, len(individual) - 1)
	rand_gen = individual[rand_index]
	rand_gen.orientation = random.choice([Orientation.Horizontal, Orientation.Vertical])
	if rand_gen.orientation == Orientation.Horizontal:
		x = random.randint(0, META.grid_size - rand_gen.length)
		y = random.randint(0, META.grid_size - 1)
		x_ = x + rand_gen.length - 1
		y_ = y
	else:
		x = random.randint(0, META.grid_size - 1)
		y = random.randint(0, META.grid_size - rand_gen.length)
		x_ = x
		y_ = y + rand_gen.length - 1
	rand_gen.position = (x, y)
	rand_gen.end = (x_, y_)
	individual[rand_index] = rand_gen
	return individual


def crossover(parent1, parent2):
	point1 = random.randint(0, len(parent1) - 1)
	offspring1 = parent1[:point1] + parent2[point1:]
	offspring2 = parent2[:point1] + parent1[point1:]
	return offspring1, offspring2


def roulette_wheel_selection(population, fitness_arr):
	fitness_sum = sum(fitness_arr)
	rand_val = random.randint(0, fitness_sum)
	prev = 0
	index = 0
	val = fitness_arr[index]
	while not (prev <= rand_val <= val):
		index += 1
		prev = val
		val += fitness_arr[index]
	return population[index]


def next_generation(population):
	fitness_arr = [fitness(individual) for individual in population]

	new_population = []
	for i in range(round(len(population) * META.crossover_coef / 2)):
		offspring1, offspring2 = crossover(roulette_wheel_selection(population, fitness_arr),
										   roulette_wheel_selection(population, fitness_arr))
		new_population.append(offspring1)
		new_population.append(offspring2)

	for j in range(round(len(population) * META.mutation_per)):
		index = random.randint(0, len(population) - 1)
		new_population.append(mutation(population[index]))

	n = round(len(population) * META.old_best_coef)
	sorted_indices = np.argsort(fitness_arr)
	result_indices = sorted_indices[-n:][::-1]
	for i in range(n):
		new_population.append(population[result_indices[i]])
	return new_population


def initialization(w: list[Word]):
	words = copy.deepcopy(w)
	population = []
	for word in words:
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
		population.append(word)
	return population


def input():
	with open("input.txt", "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


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


def best(population):
	fitness_arr = np.asarray([fitness(individual) for individual in population])
	ind = np.argmax(fitness_arr)
	print(fitness_arr[ind])
	print_crossword(population[ind])
	fitness(population[ind])


# TODO: full graph
if __name__ == '__main__':
	words = input()
	population = [initialization(words) for k in range(META.gen_size)]
	i = 0
	for gen in range(META.generations):
		population = next_generation(population)
		if i % 50 == 1:
			best(population)
		i+=1