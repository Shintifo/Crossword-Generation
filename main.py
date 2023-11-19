from enum import Enum


class META:
	grid_size = 20
	mutation_per = 0.1
	crossover_coef = 0.8
	old_best_coef = 1 - mutation_per - crossover_coef
	generations = 100
	gen_size = 1000

class Orientation(Enum):
	Horizontal = 0,
	Vertical = 1,
	NotUsed = -1

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


def input():
	with open("input.txt", "r") as file:
		return [Word(word.strip()) for word in file.readlines()]


if __name__ == '__main__':
	words = input()