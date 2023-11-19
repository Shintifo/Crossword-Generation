import random

GRID_SIZE = 20
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
MAX_GENERATIONS = 1000

word_list = ["python", "crossword", "algorithm", "evolution", "mutation", "crossover"]


def initialize_population(word_list, population_size, grid_size):
	population = []
	for _ in range(population_size):
		crossword = []
		for word in word_list:
			orientation = random.choice(["horizontal", "vertical"])
			if orientation == "horizontal":
				x = random.randint(0, grid_size - len(word))
				y = random.randint(0, grid_size - 1)
			else:
				x = random.randint(0, grid_size - 1)
				y = random.randint(0, grid_size - len(word))
			crossword.append({"word": word, "x": x, "y": y, "orientation": orientation})
		population.append(crossword)
	return population


def evaluate_fitness(crossword):
	# Fitness can be based on the number of intersections and connectedness
	intersections = 0
	connectedness = 0

	for i in range(len(crossword)):
		for j in range(i + 1, len(crossword)):
			word1 = crossword[i]
			word2 = crossword[j]

			if word1["orientation"] == "horizontal" and word2["orientation"] == "vertical":
				if word1["x"] <= word2["x"] < word1["x"] + len(word1["word"]) and \
						word2["y"] <= word1["y"] < word2["y"] + len(word2["word"]):
					intersections += 1
			elif word1["orientation"] == "vertical" and word2["orientation"] == "horizontal":
				if word2["x"] <= word1["x"] < word2["x"] + len(word2["word"]) and \
						word1["y"] <= word2["y"] < word1["y"] + len(word1["word"]):
					intersections += 1

	# Ensure all letters are connected
	connectedness = len(set((entry["x"], entry["y"]) for entry in crossword))

	# Higher fitness for more intersections and connectedness
	fitness = intersections + connectedness
	return fitness


def crossover(parent1, parent2):
	# Perform crossover by swapping words between parents
	child = []
	for word1, word2 in zip(parent1, parent2):
		child.append(word1 if random.choice([True, False]) else word2)
	return child


def mutate(crossword):
	# Perform mutation by randomly changing the position or orientation of a word
	mutated_crossword = crossword.copy()
	for i in range(len(mutated_crossword)):
		if random.random() < MUTATION_RATE:
			orientation = random.choice(["horizontal", "vertical"])
			if orientation == "horizontal":
				mutated_crossword[i]["x"] = random.randint(0, GRID_SIZE - len(mutated_crossword[i]["word"]))
				mutated_crossword[i]["y"] = random.randint(0, GRID_SIZE - 1)
			else:
				mutated_crossword[i]["x"] = random.randint(0, GRID_SIZE - 1)
				mutated_crossword[i]["y"] = random.randint(0, GRID_SIZE - len(mutated_crossword[i]["word"]))
			mutated_crossword[i]["orientation"] = orientation
	return mutated_crossword


def evolve(word_list, population_size, grid_size, mutation_rate, crossover_rate, max_generations):
	population = initialize_population(word_list, population_size, grid_size)
	for generation in range(max_generations):
		fitness_scores = [evaluate_fitness(crossword) for crossword in population]
		parents = random.choices(population, weights=fitness_scores, k=2)
		child = crossover(parents[0], parents[1]) if random.random() < crossover_rate else random.choice(parents)
		child = mutate(child) if random.random() < mutation_rate else child
		worst_index = min(range(population_size), key=lambda i: fitness_scores[i])
		population[worst_index] = child
		best_index = max(range(population_size), key=lambda i: fitness_scores[i])
		print(f"Generation {generation + 1}: Best Fitness = {fitness_scores[best_index]}")
	best_crossword = population[best_index]
	return best_crossword


class META:
	grid_size = 20


def input():
	with open("input.txt", "r") as file:
		return [word.strip() for word in file.readlines()]


word_list = input()
best_crossword = evolve(word_list, POPULATION_SIZE, GRID_SIZE, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS)

print("\nBest Crossword:")
for entry in best_crossword:
	print(f"{entry['word']} ({entry['x']}, {entry['y']}, {entry['orientation']})")
