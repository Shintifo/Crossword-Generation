import copy
import random
import numpy as np
from enum import Enum
import time

class Orientation(Enum):
	Horizontal = 0,
	Vertical = 1,
	NotUsed = -1


class META:
	grid_size = 20
	mutation_per = 0.1
	crossover_coef = 0.8
	old_best_coef = 1 - mutation_per - crossover_coef
	generations = 100
	gen_size = 1000


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.neighbours = {}  # Key - (char, num)  Value - Word
		self.orientation = Orientation.NotUsed
		self.position: tuple = None
		self.end: tuple = None

	def __lt__(self, other):
		if self.position[0] < other.position[0]:
			return True
		elif self.position[0] == other.position[0] and self.position[1] < other.position[1]:
			return True
		else:
			return False


class Cell:
	def __init__(self, word: Word = None, index: int = -1):
		self.word: Word = word
		self.index: int = index
		self.value = "-" if self.word is None else word.string[index]


class Grid:
	def __init__(self, words):
		self.words = words
		self.grid = self.init_grid()

	def init_grid(self):
		print("Todo")


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


def fitness(crossword):
	def count_penalty(word):
		timer = []

		penalty = 0
		first_neighbour = True
		num_of_inter = 0

		for word_ in crossword:
			start = time.time()
			if word_.string == word.string:
				timer.append((time.time() - start))
				continue

			intersect = find_intersection_point(word, word_)
			# print(f"	Compare word: {word_.string}")
			# print(f"		Intersection: {intersect}")

			# TODO too close, but intersect with 3 word, so it's good
			if intersect is None:
				# print(f"		Orientations the same")

				x = 1 if word.orientation == Orientation.Vertical else 0
				y = 0 if word.orientation == Orientation.Vertical else 1

				if word.position[y] == word_.position[y]:
					if (word_.position[x] <= word.position[x] <= word_.end[x]
							or word_.position[x] <= word.end[x] <= word_.end[x]):
						penalty -= word.length
						num_of_inter += 1
					# print(f"			Overlapping!")
					elif word.position[x] - 1 == word_.end[x] or word.end[x] + 1 == word_.position[x]:
						penalty -= word.length
						num_of_inter += 1
				# print(f"			Too close")
				else:
					if word.position[y] in (word_.position[y] + 1, word_.position[y] - 1):
						if (word_.position[x] <= word.position[x] <= word_.end[x]
								or word.position[x] <= word_.position[x] <= word.end[x]):
							penalty -= word.length
							num_of_inter += 1
				# print(f"			Too close")
				timer.append((time.time() - start))
				continue

			if intersect is False:
				# print(f"		No intersection")
				vert_w = word if word.orientation == Orientation.Vertical else word_
				horiz_w = word_ if word.orientation == Orientation.Vertical else word

				if vert_w.position[0] in (horiz_w.position[0] - 1, horiz_w.end[0] + 1):
					if vert_w.position[1] <= horiz_w.position[1] <= vert_w.end[1]:
						penalty -= word.length
						num_of_inter += 1
				# print(f"			Too close")
				elif horiz_w.position[1] in (vert_w.position[1] - 1, vert_w.end[1] + 1):
					if horiz_w.position[0] <= vert_w.position[0] <= horiz_w.end[0]:
						penalty -= word.length
						num_of_inter += 1
				# print(f"			Too close")
				timer.append((time.time() - start))
				continue

			# print(f"		We have an intersection!")
			vert_w = word if word.orientation == Orientation.Vertical else word_
			horiz_w = word_ if word.orientation == Orientation.Vertical else word

			if vert_w.string[intersect[1] - vert_w.position[1]] == horiz_w.string[intersect[0] - horiz_w.position[0]]:
				penalty += (word.length + word.length)*5
				num_of_inter += 1
				first_neighbour = False
			# print(f"			Wow good intersection")
			else:
				penalty -= word.length
				num_of_inter += 1
		# print(f"			Yuck bad intersection((((")

			penalty -= word.length if first_neighbour else 0  # Not Connected graph
			first_neighbour = False
			timer.append((time.time() - start))

		penalty -= abs(5 - num_of_inter)*2

		# print(f"Average per word: {sum(timer)/len(timer)}")

		return penalty

	score = 50 * len(crossword)
	for word in crossword:
		# print(f"Word: {word.string}, location: {word.position}")
		score += count_penalty(word)
	return score


def mutation(individual):
	rand_index = random.randint(0, len(individual)-1)
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
	def uniform_order():
		offspring1 = copy.deepcopy(parent1)
		offspring2 = copy.deepcopy(parent2)
		for i in range(len(offspring1)):
			if random.choice(range(1)) == 1:
				temp = offspring1[i]
				offspring1[i] = offspring2[i]
				offspring2[i] = temp
		return offspring1, offspring2

	def two_points():
		point1 = random.randint(0, len(parent1)//2)
		point2 = random.randint(len(parent1)//2+1, len(parent2))

		offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
		offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
		if len(offspring1) > 10 or len(offspring2) > 10:
			print("ABOBA")
			exit(1)
		return offspring1, offspring2

	def one_point():
		point1 = random.choice(parent1)
		offspring1 = parent1[:point1] + parent2[point1:]
		offspring2 = parent2[:point1] + parent1[point1:]
		return offspring1, offspring2

	return two_points()


def roulette_wheel_selection(population, fitness_arr):
	fitness_sum = sum(fitness_arr)
	a_bor = min(min(fitness_arr), fitness_sum)
	b_bor = max(max(fitness_arr), fitness_sum)

	rand_val = random.randint(a_bor, b_bor)
	prev = 0
	index = 0
	val = fitness_arr[index]
	while not (abs(prev) <= abs(rand_val) <= abs(val)):
		index += 1
		prev = val
		val += fitness_arr[index]
	return population[index]


# def check_position(word, population: list[Word]) -> bool:
# 	if word.position[0] < 0 or word.position[1] < 0 or (word.orientation == Orientation.Horizontal
# 														and word.position[0] + word.length > META.grid_size) or (
# 			word.orientation == Orientation.Vertical and word.position[1] + word.length > META.grid_size):
# 		return False
#
# 	direction = word.orientation
# 	x, y = word.position
# 	for entry in population:
# 		if direction == Orientation.Horizontal:
# 			if y == entry.position[1] and entry.position[0] <= x < entry.position[0] + entry.length:
# 				return False
# 		else:
# 			if x == entry.position[0] and entry.position[1] <= y < entry.position[1] + entry.length:
# 				return False
#
# 	return True


def next_generation(population):
	fitness_arr = [fitness(individual) for individual in population]

	new_population = []
	# print(round(len(population) * META.crossover_coef/2))
	# print(fitness_arr)
	for i in range(round(len(population) * META.crossover_coef/2)):
		# TODO check that parents not the same!
		offspring1, offspring2 = crossover(roulette_wheel_selection(population, fitness_arr),
										   roulette_wheel_selection(population, fitness_arr))
		new_population.append(offspring1)
		new_population.append(offspring2)
	for j in range(round(len(population) * META.mutation_per)):
		index = random.randint(0, len(population)-1)
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
		while word.orientation == Orientation.NotUsed:
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
			# if not check_position(word, population):
			# 	word.orientation = Orientation.NotUsed
			# 	word.position = None
		population.append(word)
	return population


def input():
	with open("input.txt", "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


def best(population):
	fitness_arr = np.asarray([fitness(individual) for individual in population])
	ind = np.argmax(fitness_arr)
	print(fitness_arr[ind])
	print_crossword(population[ind])

# TODO: check final condition
# TODO check max gotness at each gen
if __name__ == '__main__':
	words = input()
	population = [initialization(words) for k in range(META.gen_size)]
	i = 0
	for gen in range(META.generations):
		print(i)
		# for j in range(len(population)):
		# 	print("Variant ", j)
		# 	print_crossword(population[j])
		i+=1
		population = next_generation(population)
	# print_crossword(population)

	best(population)
