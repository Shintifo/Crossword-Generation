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
	crossover_per = 0.8
	old_best_per = 0.15

	generations = 5000
	population_size = 1000

	basic_score = 0
	penalty = 10


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.intersections: dict = {}

		self.orientation: Orientation = Orientation.NotUsed
		self.position: tuple = None
		self.end: tuple = None
		self.visited: bool = False

	def overlap(self, other) -> bool:
		par_1, par_2 = (1, 0) if self.orientation == Orientation.Vertical else (0, 1)

		if (self.position[par_1] <= other.position[par_1] <= self.end[par_1]
				or self.position[par_1] <= other.end[par_1] <= self.end[par_1]):
			if self.position[par_2] == other.position[par_2]:
				return True
		return False


def find_intersection_point(word1, word2) -> tuple[int] | bool:
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

	# for word in crossword:
	# 	for word_ in crossword:
	# 		if word_.string == word.string:
	# 			continue
	#
	# 		if word_.orientation == word.orientation:
	# 			par_1, par_2 = (1, 0) if word.orientation == Orientation.Vertical else (0, 1)
	#
	# 			if word_.position[par_2] in (word.position[par_2] + 1, word.position[par_2] - 1):
	# 				if word_.position[par_1] == word.end[par_1]:
	# 					# 		Check that both of those positions are intersections
	# 					if not word.intersections.get(word.end) or not word_.intersections.get(word_.position):
	# 						score -= META.penalty
	# 				elif word.position[par_1] == word_.end[par_1]:
	# 					if not word.intersections.get(word.position) or not word_.intersections.get(word_.end):
	# 						score -= META.penalty

	for i in range(len(crossword)):
		dfs(crossword[i])
		for word in crossword:
			if not word.visited or len(word.intersections) == 0:
				score -= META.penalty
			else:
				word.visited = False
	return score


def mutation(crossword):
	individual = copy.deepcopy(crossword)
	rand_gen = random.choice(individual)
	random_position(rand_gen)
	return individual


def crossover(parent1, parent2):
	point1, point2 = sorted(random.sample(range(1, len(parent1) - 2), 2))
	offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
	offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
	return offspring1, offspring2


def roulette_selection(population, fitness_arr):
	rand_val = random.randint(0, abs(sum(fitness_arr)))
	summation = 0
	for index, val in enumerate(fitness_arr):
		summation += abs(val)
		if rand_val <= summation:
			return population[index]


def best_old_individuals(population, fitness_arr):
	def get_n_max_indices(arr):
		n = round(len(population) * META.old_best_per)
		sorted_indices = np.argsort(arr)
		max_indices = sorted_indices[-n:]
		return max_indices[::-1]

	popul = copy.deepcopy(population)
	fit_arr = copy.deepcopy(fitness_arr)
	best_ind = [popul[i] for i in get_n_max_indices(fit_arr)]

	return best_ind


def next_generation(population, fitness_arr):
	best_old = best_old_individuals(population, fitness_arr)
	new_population = copy.deepcopy(best_old)

	for _ in range(round((len(population) * META.crossover_per) / 2)):
		# offspring1, offspring2 = crossover(random.choice(best_old), random.choice(best_old))
		offspring1, offspring2 = crossover(roulette_selection(population, fitness_arr),
										   roulette_selection(population, fitness_arr))
		new_population.append(offspring1)
		new_population.append(offspring2)

	for _ in range(round(len(population) * META.mutation_per)):
		new_population.append(mutation(random.choice(population)))

	# random.shuffle(new_population)

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


def best(population, i, fitness_arr):
	ind = np.argmax(fitness_arr)

	if i % 50 == 0:
		print(fitness_arr[ind])
		print_crossword(population[ind])

	return population[ind], fitness_arr[ind]


def dfs(word: Word):
	word.visited = True
	for word_ in word.intersections.values():
		if not word_.visited:
			dfs(word_)


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


def initialization(w: list[Word]):
	words = copy.deepcopy(w)
	for word in words:
		random_position(word)
	return words


def read_file():
	with open("input.txt", "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


def test():
	crossword = []
	word1 = Word("aboba")
	word1.orientation = Orientation.Horizontal
	word1.position = (10, 10)
	word1.end = (14, 10)

	word2 = Word("avtoza")
	word2.orientation = Orientation.Vertical
	word2.position = (10, 5)
	word2.end = (10, 10)

	word3 = Word("bolgarka")
	word3.orientation = Orientation.Vertical
	word3.position = (11, 10)
	word3.end = (11, 17)

	crossword.append(word1)
	crossword.append(word2)
	crossword.append(word3)
	print_crossword(crossword)
	print(fitness(crossword))



if __name__ == '__main__':
	words = read_file()
	population = [initialization(words) for k in range(META.population_size)]

	for i in range(META.generations):
		fitness_arr = [fitness(individual) for individual in population]
		best_crossword, best_fitness = best(population, i, fitness_arr)
		print(f"{i}, {len(population)}")
		if best_fitness == 0:
			print(f"Yay!, {i}")
			print_crossword(best_crossword)
			exit(0)
		population = next_generation(population, fitness_arr)

	print("Not successful")
