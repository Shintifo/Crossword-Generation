import copy
import random
import numpy as np
from enum import Enum


class META:
	grid_size = 40
	start_score = 1000
	basic_location = (grid_size // 2, grid_size // 2)

	mutation_per = 0.0
	crossover_coef = 0.9
	old_best_coef = 1 - mutation_per - crossover_coef

	generations = 5
	gen_size = 100


class Orientation(Enum):
	Horizontal = 0,
	Vertical = 1,
	NotUsed = -1


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.orientation = Orientation.NotUsed
		self.position: tuple = None
		self.end: tuple = None
		self.p = None
		# Keys: word, index_self, index_other
		self.intersections = []  # Contains word and position letters that intersects (for both words.txt)

	def __eq__(self, other):
		return self.string == other.string

	def set_position(self, position):
		self.position = position
		if self.orientation == Orientation.Vertical:
			self.end = tuple(map(sum, zip(self.position, (0, self.length - 1))))
		else:
			self.end = tuple(map(sum, zip(self.position, (self.length - 1, 0))))

	def intersect(self, other) -> bool:
		repeat = []
		for letter in self.string:
			word_1 = np.where(np.asarray(list(self.string)) == letter)[0]
			word_2 = np.where(np.asarray(list(other.string)) == letter)[0]
			if len(word_1) and len(word_2):
				nums_to_exclude_1 = [entity['index_self'] for entity in self.intersections]
				nums_to_exclude_2 = [entity['index_self'] for entity in other.intersections]

				entity = {'letter': letter,
						  'word_1': word_1[~np.isin(word_1, nums_to_exclude_1)],
						  'word_2': word_2[~np.isin(word_2, nums_to_exclude_2)]}
				if len(entity['word_1']) and len(entity['word_2']):
					repeat.append(entity)

		if not repeat:
			return False

		intersect_letter = random.choice(repeat)
		index1 = random.choice(intersect_letter['word_1'])
		index2 = random.choice(intersect_letter['word_2'])

		other.intersections.append({'word': self, 'index_self': index2, 'index_other': index1})
		self.intersections.append({'word': other, 'index_self': index1, 'index_other': index2})
		return True

	def parallel_neighbour(self, other) -> bool:
		par_1 = 1 * (self.orientation == Orientation.Vertical)
		par_2 = 1 * (par_1 == 0)

		if other.position[par_2] in (self.position[par_2] + 1, self.position[par_2] - 1):
			if (self.position[par_1] <= other.position[par_1] < self.end[par_1]
					or self.position[par_1] < other.end[par_1] <= self.end[par_1]):
				return True

			if other.position[par_1] == self.end[par_1]:
				if (any(self.string[entry['index_self']] == self.string[-1] for entry in self.intersections)
						and any(other.string[entry['index_self']] == other.string[1] for entry in other.intersections)):
					return False
				else:
					return True
			if self.position[par_1] == other.end[par_1]:
				if (any(self.string[entry['index_self']] == self.string[1] for entry in self.intersections)
						and any(
							other.string[entry['index_self']] == other.string[-1] for entry in other.intersections)):
					return False
				else:
					return True

		return False

	def overlap(self, other) -> bool:
		par_1 = 1 * (self.orientation == Orientation.Vertical)
		par_2 = 1 * (par_1 == 0)

		if (self.position[par_1] <= other.position[par_1] <= self.end[par_1]
				or self.position[par_1] <= other.end[par_1] <= self.end[par_1]):
			if self.position[par_2] == other.position[par_2]:
				return True
		return False

	def add_intersection_point(self, other, point_pos):
		if self.orientation == Orientation.Vertical:
			self.intersections.append({'word': other,
									   'index_self': point_pos[1] - self.position[1],
									   'index_other': point_pos[0] - other.position[0]})
			other.intersections.append({'word': self,
										'index_self': point_pos[0] - other.position[0],
										'index_other': point_pos[1] - self.position[1]})
		else:
			self.intersections.append({'word': other,
									   'index_self': point_pos[0] - self.position[0],
									   'index_other': point_pos[1] - other.position[1]})
			other.intersections.append({'word': self,
										'index_self': point_pos[1] - other.position[1],
										'index_other': point_pos[0] - self.position[0]})


def mutation(individual):
	index1 = random.randint(0, len(individual) - 1)
	index2 = random.randint(0, len(individual) - 1)
	temp = individual[index1]
	individual[index1] = individual[index2]
	individual[index2] = temp
	for word in individual:
		word.position = None
		word.intersections = []
	return individual


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


def crossover(parent1, parent2):
	offspring1 = []
	offspring2 = []

	for word in parent1:
		if random.choice([True, False]):
			offspring1.append(word)
			offspring2.append(parent2[parent2.index(word)])
		else:
			offspring2.append(word)
			offspring1.append(parent2[parent2.index(word)])

	for i in range(len(parent1)):
		offspring1[i].position = None
		offspring2[i].position = None
		offspring1[i].intersections = []
		offspring2[i].intersections = []
	return offspring1, offspring2


def mapping(individual):
	def recursion(word):
		for entity in word.intersections:
			if entity['word'].position is None:
				if word.orientation == Orientation.Vertical:
					intersection_point = tuple(map(sum, zip(word.position, (0, entity['index_self']))))
					entity['word'].set_position(tuple(map(sum, zip(intersection_point, (-entity['index_other'], 0)))))
					entity['word'].orientation = Orientation.Horizontal
				else:
					intersection_point = tuple(map(sum, zip(word.position, (entity['index_self'], 0))))
					entity['word'].set_position(tuple(map(sum, zip(intersection_point, (0, -entity['index_other'])))))
					entity['word'].orientation = Orientation.Vertical

				recursion(entity['word'])

	individual[0].set_position(META.basic_location)
	individual[0].orientation = Orientation.Vertical
	recursion(individual[0])


def find_intersection_point(word1, word2):
	x1, y1 = word1.position
	x2, y2 = word1.end
	x3, y3 = word2.position
	x4, y4 = word2.end

	if y1 == y2 and y3 == y4 or x1 == x2 and x3 == x4:
		return None

	# For Lines with One Horizontal and One Vertical
	elif (x1 == x2 and y3 == y4) or (y1 == y2 and x3 == x4):
		if x1 == x2:
			if min(x3, x4) <= x1 <= max(x3, x4) and min(y1, y2) <= y3 <= max(y1, y2):
				return x1, y3
		elif y1 == y2:
			if min(y3, y4) <= y1 <= max(y3, y4) and min(x1, x2) <= x3 <= max(x1, x2):
				return x3, y1
	# No intersection
	return False


def fitness(individual: list[Word]):
	score = META.start_score
	# Check for overlapping letters
	for i in range(1, len(individual)):
		if not individual[i].intersect(individual[i - 1]):  # If we have no intersection with previous word
			score -= individual[i].length

	# Firstly we should define the location of words.txt
	mapping(individual)
	# TODO: if no position
	x = -1, META.grid_size + 1
	y = -1, META.grid_size + 1

	for word_1 in individual:
		# Needs to check size of crosswords
		if Orientation.NotUsed == word_1.orientation:
			score -= word_1.length
			continue
		x = (max(x[0], word_1.end[0]), min(x[1], word_1.position[0]))
		y = (max(y[0], word_1.end[1]), min(y[1], word_1.position[1]))

		for word_2 in individual:
			if word_1 == word_2:
				continue
			if Orientation.NotUsed == word_2.orientation:
				score -= word_2.length
				continue

			if word_1.orientation == word_2.orientation:
				if word_1.overlap(word_2) or word_1.parallel_neighbour(word_2):  # Overlapping or neighbour
					score -= word_1.length

			else:
				vert_w = word_1 if word_1.orientation == Orientation.Vertical else word_2
				horiz_w = word_2 if word_1.orientation == Orientation.Vertical else word_1

				if vert_w.position[0] in (horiz_w.position[0] - 1, horiz_w.end[0] + 1):
					if vert_w.position[1] <= horiz_w.position[1] <= vert_w.end[1]:
						score -= word_1.length
				elif horiz_w.position[1] in (vert_w.position[1] - 1, vert_w.end[1] + 1):
					if horiz_w.position[0] <= vert_w.position[0] <= horiz_w.end[0]:
						score -= word_1.length
				else:  # Intersection
					inter_point = find_intersection_point(word_1, word_2)
					if inter_point:
						if (vert_w.string[inter_point[1] - vert_w.position[1]] ==
								horiz_w.string[inter_point[0] - horiz_w.position[0]]):
							if all(entry['index_self'] != inter_point[1] - vert_w.position[1] for entry in
								   vert_w.intersections):
								vert_w.add_intersection_point(horiz_w, inter_point)

							score += word_1.length
						else:
							score -= vert_w.length

	if any(diff > META.grid_size for diff in (x[0] - x[1], y[0] - y[1])):
		score -= META.grid_size

	return max(0, score)


def evolve(population):
	fitness_arr = [fitness(individual) for individual in population]

	next_generation = []
	for i in range(round(len(population) * META.crossover_coef // 2)):
		offspring1, offspring2 = crossover(roulette_wheel_selection(population, fitness_arr),
										   roulette_wheel_selection(population, fitness_arr))
		next_generation.append(offspring1)
		next_generation.append(offspring2)

	# for i in range(round(len(population) * META.mutation_per)):
	# 	index = random.randint(0, len(population)-1)
	# 	next_generation.append(mutation(population[index]))

	n = round(len(population) * META.old_best_coef)
	result_indices = np.argsort(fitness_arr)[-n:][::-1]
	for i in range(n):
		next_generation.append(population[result_indices[i]])

	return next_generation


def input():
	with open("input.txt", "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


def initialize(w: list[Word]):
	temp = copy.deepcopy(w)
	words = []
	for i in range(len(w)):
		index = random.randint(0, len(temp) - 1)
		words.append(temp[index])
		temp.pop(index)
	return words


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


# TODO set positions valid
# TODO reload positions
if __name__ == '__main__':
	words = input()
	population = [initialize(words) for i in range(META.gen_size)]
	for i in range(META.generations):
		print(i, len(population))
		population = evolve(population)
	fitness_arr = [fitness(individual) for individual in population]
	best = np.argmax(fitness_arr)
	print(fitness_arr[best])
	print_crossword(population[best])