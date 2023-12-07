import copy
import random
import time
from enum import Enum
import os


class Orientation(Enum):
	Horizontal = 0,
	Vertical = 1,
	NotUsed = -1,


class META:
	"""
	Class containing coefficients for EA,
	meta values, paths to input/output directories.
	"""
	mutation_per = 0.3
	crossover_per = 0.6
	old_best_per = 0.1

	grid_size = 20
	generations = 100
	population_size = 500

	basic_score = 0
	penalty = 1

	output_path = "outputs"
	input_path = "inputs"


class Word:
	def __init__(self, string: str):
		self.string = string
		self.length = len(self.string)
		self.intersections: dict = {}  # Key - location, Value - Word

		self.orientation: Orientation = Orientation.NotUsed
		self.start: tuple = None
		self.end: tuple = None
		self.visited: bool = False  # Required for DFS run

	def __lt__(self, other: 'Word') -> bool:
		return self.length > other.length

	def __eq__(self, other: str):
		return self.string == other

	def overlap(self, other: 'Word') -> bool:
		"""
		Check whether words with similar orientation overlap or not
		:param other: Word
		:return: Bool value
		"""
		# Define which axis is main for checkout
		sec_axis, main_axis = (1, 0) if self.orientation == Orientation.Vertical else (0, 1)

		return self.start[main_axis] == other.start[main_axis] and (
				self.start[sec_axis] <= other.start[sec_axis] <= self.end[sec_axis]
				or self.start[sec_axis] <= other.end[sec_axis] <= self.end[sec_axis]
		)

	def are_intersect(self, other: 'Word') -> bool | tuple[int]:
		"""
		Finds point of intersection of two orthogonal words, if it exists.
		:param other: word.
		:return:Point of intersection or bool value of its absence.
		"""
		x1, y1 = self.start
		x2, y2 = self.end
		x3, y3 = other.start
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
		"""
		Randomly chooses the word orientation and position,
		taking into consideration word's length.
		"""
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
		self.start = (x, y)
		self.end = (x_, y_)
		return self


class Crossword:
	def __init__(self, words):
		self.words: list[Word] = words
		self.words_num = len(self.words)

	def dfs(self, word: Word):
		"""
		Standard recursive DFS algorithm
		:param word: Start word
		"""
		word.visited = True
		for word_ in word.intersections.values():
			if not word_.visited:
				self.dfs(word_)

	def fitness(self) -> int:
		"""
		Measure how fit an individual(crossword) is.
		:return: Crossword score
		"""
		crossword = copy.deepcopy(self.words)
		score = META.basic_score  # score == 0
		# Compare all words with each other
		for word_1 in crossword:
			for word_2 in crossword:
				if word_2.string == word_1.string:
					continue

				if word_2.orientation == word_1.orientation:
					par_1, par_2 = (1, 0) if word_1.orientation == Orientation.Vertical else (0, 1)

					if word_1.overlap(word_2):
						score -= META.penalty
					elif word_2.start[par_2] in (word_1.start[par_2] + 1, word_1.start[par_2] - 1):
						if (word_1.start[par_1] <= word_2.start[par_1] < word_1.end[par_1]
								or word_1.start[par_1] < word_2.end[par_1] <= word_1.end[par_1]):
							score -= META.penalty
					elif word_2.start[par_2] == word_1.start[par_2]:
						if word_2.end[par_1] == word_1.start[par_1] - 1 or word_2.start[par_1] == word_1.end[par_1] + 1:
							score -= META.penalty
				else:
					intersect = word_1.are_intersect(word_2)
					vert_w = word_1 if word_1.orientation == Orientation.Vertical else word_2
					horiz_w = word_2 if word_1.orientation == Orientation.Vertical else word_1

					if intersect:
						if (vert_w.string[intersect[1] - vert_w.start[1]]
								!= horiz_w.string[intersect[0] - horiz_w.start[0]]):
							score -= META.penalty
						else:
							word_1.intersections[intersect] = word_2
					else:
						if vert_w.start[0] in (horiz_w.start[0] - 1, horiz_w.end[0] + 1):
							if vert_w.start[1] <= horiz_w.start[1] <= vert_w.end[1]:
								score -= META.penalty
						elif horiz_w.start[1] in (vert_w.start[1] - 1, vert_w.end[1] + 1):
							if horiz_w.start[0] <= vert_w.start[0] <= horiz_w.end[0]:
								score -= META.penalty

		# Check the case of parallel neighbour words, for first and last symbols in the words
		for word_1 in crossword:
			for word_2 in crossword:
				if word_2.string == word_1.string or word_2.orientation != word_1.orientation:
					continue

				par_1, par_2 = (1, 0) if word_1.orientation == Orientation.Vertical else (0, 1)

				if word_2.start[par_2] in (word_1.start[par_2] + 1, word_1.start[par_2] - 1):
					if word_2.start[par_1] == word_1.end[par_1]:
						if not word_1.intersections.get(word_1.end) or not word_2.intersections.get(word_2.start):
							score -= META.penalty
					elif word_1.start[par_1] == word_2.end[par_1]:
						if not word_1.intersections.get(word_1.start) or not word_2.intersections.get(word_2.end):
							score -= META.penalty

		# Checks consistency of the crossword using DFS
		for i in range(self.words_num):
			self.dfs(crossword[i])
			for word_1 in crossword:
				if not word_1.visited or len(word_1.intersections) == 0:
					score -= META.penalty
				else:
					word_1.visited = False
		return score

	def mutation(self):
		"""
		Randomly mutates one word in crossword.
		"""
		new_word_set = copy.deepcopy(self.words)
		rand_gen = random.choice(new_word_set)
		rand_gen.random_position()
		return Crossword(new_word_set)

	def crossover(self, parent2: 'Crossword') -> tuple[Word]:
		"""
		Cross genes of two parent using one point or two point strategy.
		If the total amount of words < 3, perform one point crossover.
		:param parent2: Crossword
		:return: Two offsprings(crosswords)
		"""

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
		"""
		Print the crossword into console
		"""
		grid = []
		for _ in range(META.grid_size):
			grid.append(['-' for _ in range(META.grid_size)])
		for word in self.words:
			x, y = word.start
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
		"""
		Insert new word into crossword
		:param word: New word
		"""
		self.words.append(word)
		self.words_num += 1

	def to_output(self) -> str:
		"""
		Prepare the output to the file.
		:return: String for the output file.
		"""
		out_string = ""
		for word in self.words:
			out_string += f"{word.start[0]} {word.start[1]} {word.orientation.value[0]}\n"
		return out_string


def best_old_individuals(population: list[Crossword], fitness_arr: list[int]) -> list[Crossword]:
	"""
	Return best 30% individuals of the population.
	:param population: Population of crosswords.
	:param fitness_arr: Fitness fo each crossword.
	:return: List of the best crosswords.
	"""

	def get_max_indices(arr):
		"""
		Finds best fitness values.
		:param arr: Fitness array.
		:return: Array of indexes of the best fitness values.
		"""
		n = round(len(pop) * 0.3)
		sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k])
		max_indices = sorted_indices[-n:][::-1]
		return max_indices

	pop = copy.deepcopy(population)
	fit_arr = copy.deepcopy(fitness_arr)
	return [pop[i] for i in get_max_indices(fit_arr)]


def evolve(population: list[Crossword], fitness_arr: list[int]) -> list[Crossword]:
	"""
	Evolve the population.
	:param population: Current population.
	:param fitness_arr: Fitness values of current population.
	:return: New population
	"""
	# Choose the best individual of current population
	best_old_individual = best_old_individuals(population, fitness_arr)
	# Take certain percentage of the best individual to the next generation.
	n = round(len(population) * META.old_best_per)
	new_generation = copy.deepcopy(best_old_individual[:n])

	# Cross the genes among the best individuals randomly
	for _ in range(round((len(population) * META.crossover_per) / 2)):
		offspring1, offspring2 = random.choice(best_old_individual).crossover(random.choice(best_old_individual))
		new_generation.append(offspring1)
		new_generation.append(offspring2)

	# Randomly mutate individuals of current generation
	for _ in range(round(len(population) * META.mutation_per)):
		new_generation.append(random.choice(population).mutation())

	return new_generation


def completed_crosswords(population: list[Crossword], fitness_arr: list[int]) -> list[Crossword]:
	"""
	Find the correct crosswords out of the population.
	:param population: Current generation.
	:param fitness_arr: Fitness values of each individual of current crossword.
	:return: Crosswords that has fitness score 0
	"""
	return [population[i] for i in range(len(fitness_arr)) if fitness_arr[i] == 0]


def has_completed_crossword(fit_arr: list[int]) -> bool:
	"""
	Checks whether population has at least one crossword with fitness score 0 or not
	:param fit_arr: Fitness scores of population
	:return: Bool value
	"""
	return fit_arr[fit_arr.index(max(fit_arr))] == 0


def init_zero_population(input_words: list[str]) -> list[Crossword]:
	"""
	Initialize zero population.
	:param input_words: Set of given words.
	:return: Initial population.
	"""
	zero_population = []
	for _ in range(META.population_size):
		words = [Word(w) for w in input_words]
		for word in words:
			word.random_position()
		zero_population.append(Crossword(words=words))
	return zero_population


def create_new_population(best_crossword: list[Crossword], new_word: str):
	"""
	Create new population of completed crossword with new word.
	:param best_crossword: Completed crossword with fewer words.
	:param new_word: Word to add to the crossword.
	:return: New population, with increased number of words.
	"""
	new_population = []
	for cross in best_crossword:
		for _ in range(META.population_size // len(best_crossword)):
			crossword = copy.deepcopy(cross)
			word = Word(new_word).random_position()
			crossword.insert_word(word)
			new_population.append(crossword)
	for i in range(META.population_size - len(new_population)):
		crossword = copy.deepcopy(random.choice(best_crossword))
		word = Word(new_word).random_position()
		crossword.insert_word(word)
		new_population.append(crossword)

	return new_population


def output_solution(file_num, crossword: Crossword = None, words_order=None):
	with open(f"{META.output_path}/output{file_num}.txt", "w") as out_file:
		if crossword is None:
			out_file.write("Fail\n")
		else:
			for word in words_order:
				cr_word = crossword.words[(crossword.words.index(word))]
				out_file.write(f"{cr_word.start[0]} {cr_word.start[1]} {cr_word.orientation.value[0]}\n")


def read_file(file_name) -> tuple[list[str], list[str]]:
	with open(f"{META.input_path}/{file_name}", "r") as input_file:
		words = [word.strip() for word in input_file.readlines()]
		input_order_words = copy.copy(words)
		words.sort(key=lambda x: len(x))
		return input_order_words, words[::-1]


def solve(file_name):
	"""
	Construct the crossword using EA algorithm.
	:param file_name: Input file
	:return: Bool
	"""
	file_num = file_name.split(".")[0][5:]

	input_order_words, words = read_file(file_name)  # Sorting words by its length(Long >> Short)
	print(f"Number of words: {len(words)}")
	# Try to construct the crossword with several attempts
	attempts = 0
	while attempts != 10:
		# Start constructing the crossword out of two words.
		used_words = 2
		initial_words_set = copy.copy(words[:used_words])
		population = init_zero_population(initial_words_set)
		fitness_arr = [individual.fitness() for individual in population]
		for generation in range(META.generations):
			population = evolve(population, fitness_arr)
			fitness_arr = [individual.fitness() for individual in population]
			if has_completed_crossword(fitness_arr):
				completed_crosswords = completed_crosswords(population, fitness_arr)
				# Checks whether we used all given words or not.
				if len(words) == population[0].words_num:
					output_solution(file_num, completed_crosswords[0], input_order_words)
					print("Successfully!")
					completed_crosswords[0].print()
					return True
				# Creates new population out of the valid crossword and next word
				population = create_new_population(completed_crosswords, words[used_words])
				used_words += 1

		attempts += 1
	print("Fail")
	output_solution(file_num)
	return False


def print_time(time_in_sec):
	sec = str(int(time_in_sec) % 60)
	if len(sec) == 1:
		sec = "0" + sec
	print(f"Time: {int(time_in_sec // 60)}:{sec}", end="\n\n\n")


if __name__ == "__main__":
	for file in os.listdir(META.input_path):
		timer = time.time()
		solve(file)
		print_time(time.time() - timer)
