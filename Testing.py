import copy
import os
import random
import shutil
import string
import matplotlib.pyplot as plt
import numpy as np

from TimofeyBrayko import solve


def generate_random_word(length):
	letters = string.ascii_lowercase
	random_word = ''.join(random.choice(letters) for _ in range(length))
	return random_word


def generator():
	with open("words.txt", 'r') as file:
		words = [word.strip() for word in file.readlines()]

	for file_num in range(25, 100):
		path = f"test_inputs/input{file_num}.txt"
		file_data = []
		created = False
		while not created:
			words_num = random.randint(5, 20)
			with open(path, 'w') as file:
				for word in range(words_num):
					flag = True

					while flag:
						rand_word = random.choice(words)
						if rand_word not in file_data:
							flag = False
							file_data.append(rand_word)
							file.write(f"{rand_word}\n")

			if solve(path, file_num):
				shutil.move(path, f"inputs/input{file_num}.txt")
				created = True


def check_similarity():
	def check_duplicates(file_path, i):
		flag = False
		word_set = set()
		with open(file_path, 'r') as file:
			for line_number, line in enumerate(file, start=1):
				words = line.strip().split()

				for word in words:
					if word in word_set:
						flag = True
						print(f"Duplicate word '{word}' found in file {i}")
					else:
						word_set.add(word)

	for i in range(0, 25):
		name = f"inputs/input{i}.txt"
		check_duplicates(name, i)


def cases():
	for i in range(0, 25):
		arr = []
		name = f"input{i}.txt"
		if name in os.listdir("inputs"):
			with open(f"inputs/{name}", "r") as file:
				for word in file.readline():
					arr.append(word.strip().lower())

			with open(f"inputs/{name}", "w") as file:
				for word in arr:
					file.write(word)


def find():
	for i in os.listdir("inputs"):
		with open(f"inputs/{i}", "r") as file:
			words = [word.strip() for word in file.readlines()]
			if len(words) > 17:
				for word in words:
					print(word)
			print("Next")


def avg_words():
	sums = 0
	for inp in os.listdir("inputs"):
		with open(f"inputs/{inp}", "r") as file:
			sums += len([word.strip() for word in file.readlines()])
	print(sums / len(os.listdir("inputs")))


def build():
	np.random.seed(seed=0)
	w_5 = [45, 52, 54, 13, 253, 12, 43]
	w_10 = [90, 92, 192, 82, 12, 49, 23, 4, 3, 3]
	a = (w_5, w_10)

	fig, ax = plt.subplots()
	ax.boxplot(a, vert=True, showmeans=False, meanline=False,
			   labels=('5', '10'))
	plt.show()


def sort_test():
	lox = ["a", "aaboba", "kur", "zadvdzavdvAvdv"]
	lox.sort(key=lambda x: len(x))
	print(lox[::-1])


def copy_test():
	lox = ["a", "aaboba", "kur", "zadvdzavdvAvdv"]
	a = copy.copy(lox)
	a[0] = "b"
	print(lox)
	print(a)


# TODO change cases in inputs
# TODO Check that input doesn't contains duplicates
if __name__ == '__main__':
	build()