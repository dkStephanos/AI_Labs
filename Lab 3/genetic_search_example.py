import os
import operator
import math
import random
import time
from copy import deepcopy


# ==============================================================================
# Goal: Show how different search methods progress through a grid
# ==============================================================================

CLEAR = "CLS"   # Clear for Windows
#CLEAR = "clear" # Clear for Linux and macOS
DISPLAY_RATE = 10      # Show a graph at this rate
ALL_RESULTS = "Results for Genetic Search Algorithm Iterations\n-------------------------\n\n"
#===============================================================================
# Method: method_timing
#  Purpose: A timing function that wraps the called method with timing code.
#     Uses: time.time(), used to determine the time before an after a call to
#            func, and then returns the difference.
def method_timing(func):
	def wrapper(*arg):
		t1 = time.time()
		res = func(*arg)
		t2 = time.time()
		print ('%s took %0.3f ms' % (func, (t2-t1)*1000.0))
		global ALL_RESULTS
		ALL_RESULTS += '%s took %0.3f ms\n\n' % (func, (t2-t1)*1000.0)
		return [res,(t2-t1)*1000.0]
	return wrapper




#===============================================================================
# Class: Field
#  Purpose: Represents a Search Field
class Field:

	def __init__(self, size):
		self.size = size
		self.board = [["." for _ in range(size)] for _ in range(size)]
		self.current = deepcopy(self.board)


	def size(self):
		return self.size


	def set_start(self,row,col):
		self.start = (row,col)
		self.board[row][col] = 'S'
		self.set_current((row,col),'S')


	def set_end(self,row,col):
		self.end = (row,col)
		self.board[row][col] = 'E'
		self.set_current((row,col),'E')


	def get_start(self):
		return self.start


	def get_end(self):
		return self.end


	def reset_current(self):
		self.current = deepcopy(self.board)


	def set_current(self,state,string):
		self.current[state[0]][state[1]] = string


	def get_current(self,state):
		return self.current[state[0]][state[1]]


	def to_s(self):
		string = ""
		for row in self.board:
			for col in row:
				string += col + " "
			string += '\n'
		return string


	def to_s_current(self):
		'''
			Returns a string representation of the current state
		'''
		string = ""
		for row in self.current:
			for col in row:
				string += col + " "
			string += '\n'
		return string



#===============================================================================
# Class: GeneticSearcher
#  Purpose: Searches a Field using a GA, attempting to minimize the fitness
#            function, which is distance traveled.
class GeneticSearcher:

	def __init__(self, field, generations, population_size, mutation_rate):
		'''
		Initializes the Genetic Searcher

		Parameters
		----------
		field : Field
			The field of dots to use
		generations : int
			The number of generations to employ
		population_size : int
			The number of individuals in each generation
		mutation_rate : double
			The mutation rate

		Returns
		-------
		None.

		'''
		self.field = field
		self.chromosome_size = field.size * field.size # Field is size X size
		self.generations = generations
		self.population_size = population_size
		self.mutation_rate = mutation_rate
		self.population = []
		self.generation_list = []
		self.PARENT = {1:5,2:6,3:7,4:8,5:1,6:2,7:3,8:4}


	def reset(self):
		'''
		Reset the search field.

		Returns
		-------
		None.

		'''
		self.field.reset_current()


	def get_parent_step(self, step):
		'''
		Returns the parent step of the current step

		Parameters
		----------
		step : int
			The integer representation of the current step

		Returns
		-------
		int
			The step of the parent's location

		'''		
		return self.PARENT[step]


	def get_move(self, current_in, step):
		'''
		Returns the move from current_in, based on the step direction

		Parameters
		----------
		current_in : tuple in (x,y) format
			The current location's coordinates within the grid
		step : int
			The direction to step.

		Returns
		-------
		current : tuple in (x,y) format
			The tuple that results after taking the step

		'''
		current = None
		if step == 1:
			current = (current_in[0]-1,current_in[1])
		elif step == 2:
			current = (current_in[0]-1,current_in[1]+1)
		elif step == 3:
			current = (current_in[0], current_in[1]+1)
		elif step == 4:
			current = (current_in[0]+1, current_in[1]+1)
		elif step == 5:
			current = (current_in[0]+1, current_in[1])
		elif step == 6:
			current = (current_in[0]+1, current_in[1]-1)
		elif step == 7:
			current = (current_in[0], current_in[1]-1)
		elif step == 8:
			current = (current_in[0]-1, current_in[1]-1)

		return current


	def print_chromosome(self, cf):
		'''
		Prints the chromosome to the screen as a field representation.

		Parameters
		----------
		cf : list in format [chromosome, fitness]
			A list containing the chromosome to display, and its fitness

		Returns
		-------
		None.

		'''
		# Setup the display and the variables
		self.reset()
		chromosome = cf[0]
		fitness = cf[1]
		current = self.field.get_start()
		
		# Step through the chromosome
		for step in chromosome:

			current = self.get_move(current, step)
			
			## Collision detection
			if self.out_of_bounds(current):
				break
			
			## Goal detection
			if current == self.field.get_end():
				break

			self.field.set_current(current,"X")
		
		# Display the field on the screen, along with the fitness.
		os.system(CLEAR)
		print(f"{self.field.to_s_current()}\nFitness: {fitness}")

	def fitness(self,chromosome):
		'''
		Calculates the fitness of the given chromosome

		Parameters
		----------
		chromosome : list of ints
			A list of the sequential steps in the field

		Returns
		-------
		fitness : int
			The overall fitness of the chromosome

		'''
		# Setup the fitness and starting location
		fitness = 0
		current = self.field.get_start()

		# Step through the chromosome
		for step in chromosome:

			## Reset the current location
			current = self.get_move(current, step)

			## Fitness gets worse if you have a collision
			if self.out_of_bounds(current):
				fitness = fitness + 500
				break

			## Fitness gets better if you reach the end
			if current == self.field.get_end():
				fitness = fitness - 500
				break
			
			## Record the step taken
			fitness = fitness + 1

		return fitness


	def out_of_bounds(self, move):
		'''
		Check to see if a move is outside the boundaries of the field

		Parameters
		----------
		move : tuple in the form (x,y)
			The location within the field in an x,y format

		Returns
		-------
		Boolean
			True if out of bounds, or False if in bounds

		'''
		low = 0
		high = self.field.size - 1

		return move[0] < low or move[1] < low or move[0] > high or move[1] > high


	def build_chromosome(self):
		'''
		Builds a chromosome by stepping through the field.

		Returns
		-------
		chromosome : list of int
			Returns the list of steps taken

		'''
		# Setup the chromosome and staring location.
		chromosome = []
		current = self.field.get_start()
		last = 0
		
		# Create steps from 0..chromosome_size
		for i in range(self.chromosome_size):
			step_to = random.randint(1,8)
			move = self.get_move(current,step_to)
			
			## Collision Detection
			while self.out_of_bounds(move):
				step_to = random.randint(1,8)
				move = self.get_move(current,step_to)

			## Avoid backtracking as much as possible
			while step_to == last:
				step_to = random.randint(1,8)
				move = self.get_move(current,step_to)
				while self.out_of_bounds(move):
					step_to = random.randint(1,8)
					move = self.get_move(current,step_to)
			current = move
			last = self.get_parent_step(step_to)

			chromosome.append(step_to)

		return chromosome


	def initialize_population(self):
		'''
		Creates the first set of steps (the first generation) in a controlled random manner

		Returns
		-------
		None.

		'''
		for i in range(self.population_size):

			chromosome = self.build_chromosome()

			individual = [chromosome, self.fitness(chromosome)]

			self.population.append(individual)

		self.population.sort(key=operator.itemgetter(1),reverse=False)

		self.generation_list.append(self.population)


	# Set rand to True to divert typical functionality and choose parents completely at random
	def selection(self, population, rand=False):
		'''
		Selects parents from the given population, assuming that the population is
		sorted from best to worst fitness.

		Parameters
		----------
		population : list of lists
			Each item in the population is in the form [chromosome,fitness]

		Returns
		-------
		parent1 : list of int
			The chromosome chosen as parent1
		parent2 : list of int
			The chromosome chosen as parent2

		'''
		# Set the elitism factor and calculate the max index
		if rand == False:
			factor = 0.5	# Select from top 50%
			high = math.ceil(self.population_size*factor)
		else:
			high = self.population_size - 1

		# Choose parents randomly
		parent1 = population[random.randint(0,high)][0]
		parent2 = population[random.randint(0,high)][0]

		# If the same parent is chosen, pick another
		# we can get stuck here if we converge early, if we pick the same parent ten times in a row, just bail out
		count = 0
		while str(parent1) == str(parent2):
			parent2 = population[random.randint(0,high)][0]
			count += 1
			if count == 10:
				break

		return parent1, parent2


	# Set reproduction_type to "singlepoint"/"multipoint" to divert from typical behavior and instead perform a singlepoint/multipoint reproduction strategy
	def reproduce(self, parent1, parent2, reproduction_type="uniform"):
		'''
		Uses the Uniform Crossover method to reproduce with parent1 and parent2

		Parameters
		----------
		parent1 : list of int
			A chromosome that lists the steps to take
		parent2 : list of int
			A chromosome that lists the steps to take

		Returns
		-------
		list in the form [chromosome,fitness]
			The child chromosome and its fitness value

		'''
		# Initialization
		child = []
		if reproduction_type == "singlepoint":
			# Randomly choose a split point
			split_point = self.chromosome_size - random.randint(0, self.chromosome_size)
			child = parent1[:split_point] + parent2[split_point:]
		elif reproduction_type == "multipoint":
			points = []
			while len(points) < 2: 
				split_point = self.chromosome_size - random.randint(0, self.chromosome_size) 
				if split_point not in points:
					points.append(split_point)
			points.sort()
			child = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
		else:
			# Step through each item in the chromosome and randomly choose which
			#  parent's genetic material to select
			for i in range(self.chromosome_size):
				bit = None
				if random.randint(0,1) == 0:
					bit = parent1[i]
				else:
					bit = parent2[i]
				child.append(bit)

		return [child, self.fitness(child)]


	def mutate(self, chromosome):
		'''
		Perform a standard flip mutation on the chromosome

		Parameters
		----------
		chromosome : list of int
			A list of the steps taken

		Returns
		-------
		list in the form [chromosome,fitness]
			The mutated chromosome along with its new fitness value

		'''
		ch1 = deepcopy(chromosome)
		flip = random.randint(0,len(ch1)-1)
				
		new_step = random.randint(1,8)
		
		ch1[flip] = new_step
		
		return [ch1,self.fitness(ch1)]


	@method_timing
	def run(self):
		'''
		Runs the genetic algorithm

		Returns
		-------
		None.

		'''
		# Initialize and print the initial population
		self.initialize_population()
		self.print_chromosome(self.population[0])
		global ALL_RESULTS
		generation = 1
		lowest_generation = 0
		lowest_fitness = 9999

		# Create generations through time
		while generation <= self.generations:
			new_population = []

			## Ensure you keep the best of the best from the previous generation
			retain = math.ceil(self.population_size*0.025)
			new_population = self.generation_list[generation - 1][:retain]

			## Conduct selection, reproduction, and mutation operations to fill the rest of the population
			while len(new_population) < self.population_size:
				parent1, parent2 = self.selection(self.generation_list[generation - 1])

				child = self.reproduce(parent1, parent2, "multipoint")

				if (random.random() < self.mutation_rate):
					child = self.mutate(child[0])
					
				new_population.append(child)

			generation = generation + 1

			## Sort the population in ascending order according to fitness (Low is good).
			new_population.sort(key=operator.itemgetter(1),reverse=False)
			
			low_fitness = new_population[0][1]
			if low_fitness < lowest_fitness:
				lowest_fitness = low_fitness
				lowest_generation = generation
			
			## Add the new generation and display if at the appropriate rate
			self.generation_list.append(new_population)
			if generation % DISPLAY_RATE == 0:
				self.print_chromosome(new_population[0])
				print("Generation",generation,"Fitness",new_population[0][1],"\n") 

		ALL_RESULTS += f"Lowest Fitness: {lowest_fitness} Reached at Generation: {lowest_generation}\n\n"
		print("Lowest Fitness: ",lowest_fitness,"Reached at Generation:",lowest_generation,"\n\n")




#=====================
# Main Algorithm

for itteration in range(0,10):
	num_generations = random.randint(50,150)
	population_size = random.randint(5,40)
	mutation_rate = random.randint(1,100)*.01
	ALL_RESULTS += f"----------------------------------------\n\nNext Itterations: Params = {{generations: {num_generations}, population_size: {population_size}, mutation_rate: {mutation_rate}}}\n----------------------------------------\n\n"
	ALL_RESULTS += "Results for field: 10x10:\n"
	field = Field(10)
	field.set_start(0,0)
	field.set_end(9,9)
	searcher = GeneticSearcher(field, 100, 20, 0.05)
	searcher.run()

	ALL_RESULTS += "Results for field: 20x20:\n"
	field = Field(20)
	field.set_start(0,0)
	field.set_end(19,19)
	searcher = GeneticSearcher(field, 100, 20, 0.05)
	searcher.run()

	ALL_RESULTS += "Results for field: 30x30:\n"
	field = Field(30)
	field.set_start(0,0)
	field.set_end(29,29)
	searcher = GeneticSearcher(field, 100, 20, 0.05)
	searcher.run()

text_file = open("GeneticSearchAlgorithmResults.txt", "w")
text_file.write(ALL_RESULTS)
text_file.close()