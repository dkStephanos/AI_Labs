import os
import math
import numpy as np
import time
import random
from collections import deque
from copy import deepcopy
from operator import itemgetter

# ==============================================================================
# Goal: Show how different search methods progress through a grid
# ==============================================================================

#CLEAR = "CLS"   # Clear for Windows
CLEAR = "clear" # Clear for Linux and macOS

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
		return [res,(t2-t1)*1000.0]
	return wrapper

#===============================================================================
# Class: PriorityQueue
#  Purpose: A simplified PriorityQueue
class PriorityQueue:
	def __init__(self):
		self.queue = []

	def set_priority(self, item, priority):
		for node in self.queue:
			if node[0] == item:
				self.queue.remove(node)
				break
		self.put(item, priority)

	def put(self, item, priority):
		node = [item,priority]
		self.queue.append(node)
		self.queue.sort(key=itemgetter(1))

	def get(self):
		if len(self.queue) == 0:
			return None
		node = self.queue.pop(0)
		return node[0]

	def empty(self):
		return len(self.queue) == 0


#===============================================================================
# Class: Field
#  Purpose: Represents a Search Field
class Field:

	def __init__(self, size):
		'''
			Default Constructor
		'''
		self.size = size
		self.board = [["." for _ in range(size)] for _ in range(size)]
		self.current = deepcopy(self.board)

		# Randomly add up to 10 obstacles
		for i in range(0,10):
			self.current[random.randint(0, self.size - 1)][random.randint(0, self.size - 1)] = '*'

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

	def reset_current(self):
		self.current = deepcopy(self.board)
		# Randomly add up to 10 obstacles
		for i in range(0,10):
			self.current[random.randint(0, self.size - 1)][random.randint(0, self.size - 1)] = '*'

	def set_current(self,state,string):
		self.current[state[0]][state[1]] = string

	def to_s(self):
		'''
			Returns a string representation of the Board
		'''
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
# Class: Searcher
#  Purpose: Searches a Field
class Searcher:

	def __init__(self,field):
		'''
			Constructor: accepts a field
		'''
		self.field = field

	def reset(self):
		self.field.reset_current()


	def heuristic(self,state):
		'''
			Because we cannot move diagonally, here, the heuristic h(n) is the
			 Manhattan (city block) distance. State is a tuple in (row,col) form
		'''
		return (abs(state[0]-self.field.end[0]) + abs(state[1]-self.field.end[1]))

	def diagonal_distance(self,state):
		'''
			Used to illustrate A*Search from a different perspective.
		'''
		D = 1
		D2 = math.sqrt(2)
		dx = abs(state[0] - self.field.end[0])
		dy = abs(state[1] - self.field.end[1])
		return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


	def get_neighbors(self,state):
		'''
			Returns a list of valid neighbors for the current position (state),
			  which is a tuple in the form (row,col). Since you cannot Moves
			  diagonally, you must move either N, S, E, or W.
		'''
		neighbors = []      #This is going to be a list of tuples in the form (row,col)

		left = (state[0],state[1]-1)
		right = (state[0],state[1]+1)
		up = (state[0]-1,state[1])
		down = (state[0]+1,state[1])

		# Only append the neighbor if the value is within the scope of the field
		if left[0] >= 0 and left[0] < self.field.size and \
		   left[1] >= 0 and left[1] < self.field.size:
			neighbors.append(left)
		if right[0] >= 0 and right[0] < self.field.size and \
		   right[1] >= 0 and right[1] < self.field.size:
			neighbors.append(right)
		if up[0] >= 0 and up[0] < self.field.size and \
		   up[1] >= 0 and up[1] < self.field.size:
			neighbors.append(up)
		if down[0] >= 0 and down[0] < self.field.size and \
		   down[1] >= 0 and down[1] < self.field.size:
			neighbors.append(down)

		return neighbors

	def get_diagonal_neighbors(self,state):

		neighbors = []

		rows = np.arange(state[0]-1,state[0]+2)
		rows = rows[rows >= 0]
		rows = rows[rows <= self.field.size-1]
		cols = np.arange(state[1]-1,state[1]+2)
		cols = cols[cols >= 0]
		cols = cols[cols <= self.field.size-1]

		all_neighbors = np.transpose([np.tile(rows, len(cols)), np.repeat(cols, len(rows))])

		for loc in all_neighbors:
			neighbors.append((loc[0],loc[1]))
		if state in neighbors:
			neighbors.remove(state)
		return (neighbors)

	@method_timing
	def breadth_first(self):
		'''
			Performs the breadth first search on the current field
		'''
		frontier = deque()
		explored = []

		frontier.appendleft(self.field.start)

		while True:
			if len(frontier) == 0:
				return -1
			current = frontier.pop()
			self.field.set_current(current,"X")


			for child in self.get_neighbors(current):

				if child not in frontier and child not in explored and self.field.current[child[0]][child[1]] != '*':
					if child == self.field.end:
						return 1    #Success
					frontier.appendleft(child)
					self.field.set_current(child,"C")

			os.system(CLEAR)
			print("BREADTH-FIRST SEARCH - NO DIAGONALS ALLOWED")
			print()
			print(self.field.to_s_current())
			explored.append(current)
			time.sleep(0.01)

	@method_timing
	def breadth_first_diagonal(self):
		'''
			Performs the breadth first search on the current field,
			allowing diagonal movements
		'''
		frontier = deque()
		explored = []

		frontier.appendleft(self.field.start)

		while True:
			if len(frontier) == 0:
				return -1
			current = frontier.pop()
			self.field.set_current(current,"X")


			for child in self.get_diagonal_neighbors(current):

				if child not in frontier and child not in explored and self.field.current[child[0]][child[1]] != '*':
					if child == self.field.end:
						return 1    #Success
					frontier.appendleft(child)
					self.field.set_current(child,"C")

			os.system(CLEAR)
			print("BREADTH-FIRST SEARCH - DIAGONALS ALLOWED")
			print()
			print(self.field.to_s_current())
			explored.append(current)
			time.sleep(0.01)

	@method_timing
	def depth_first(self):
		'''
			Performs the depth first search on the current field,
			NOT allowing diagonal movements
		'''
		frontier = deque()
		explored = []

		frontier.append(self.field.start)

		while True:
			if len(frontier) == 0:
				return -1
			current = frontier.pop()
			self.field.set_current(current,"X")


			for child in self.get_neighbors(current):

				if child not in frontier and child not in explored and self.field.current[child[0]][child[1]] != '*':
					if child == self.field.end:
						return 1    #Success
					frontier.append(child)
					self.field.set_current(child,"C")

			os.system(CLEAR)
			print("DEPTH-FIRST SEARCH - NO DIAGONALS ALLOWED")
			print()
			print(self.field.to_s_current())
			explored.append(current)
			time.sleep(0.01)

	@method_timing
	def depth_first_diagonal(self):
		'''
			Performs the depth first search on the current field,
			allowing diagonal movements
		'''
		frontier = deque()
		explored = []

		frontier.append(self.field.start)

		while True:
			if len(frontier) == 0:
				return -1
			current = frontier.pop()
			self.field.set_current(current,"X")


			for child in self.get_diagonal_neighbors(current):

				if child not in frontier and child not in explored and self.field.current[child[0]][child[1]] != '*':
					if child == self.field.end:
						return 1    #Success
					frontier.append(child)
					self.field.set_current(child,"C")

			os.system(CLEAR)
			print("DEPTH-FIRST SEARCH - DIAGONALS ALLOWED")
			print()
			print(self.field.to_s_current())
			explored.append(current)
			time.sleep(0.01)

	@method_timing
	def uniform_cost(self):
		'''
			Performs the uniform cost search on the current field
		'''
		frontier = PriorityQueue()
		cost_so_far = {}
		start= self.field.start
		explored = []

		frontier.put(start,0)
		cost_so_far[start] = 0

		while True:
			if frontier.empty():
				return -1
			current = frontier.get()
			if current == self.field.end:
				break
			explored.append(current)

			self.field.set_current(current,"X")

			for child in self.get_neighbors(current):

				new_cost = cost_so_far[current] + 1 # You took one step past current

				if child not in explored and self.field.current[child[0]][child[1]] != '*' and child not in cost_so_far.keys():
					if child == self.field.end:
						return 1    #Success
					frontier.put(child, new_cost)
					cost_so_far[child] = new_cost
					self.field.set_current(child,"C")

				if child in cost_so_far.keys():
					if new_cost < cost_so_far[child]:
						cost_so_far[child] = new_cost
						frontier.set_priority(child,new_cost)
						self.field.set_current(child,"R")	#Show when the item is reset
			os.system(CLEAR)
			print("UNIFORM COST SEARCH")
			print()
			print(self.field.to_s_current())
			time.sleep(0.01)

	@method_timing
	def uniform_cost_diagonal(self):
		'''
			Performs the uniform cost search on the current field
			and allows diagonal movements
		'''
		frontier = PriorityQueue()
		cost_so_far = {}
		start= self.field.start
		explored = []

		frontier.put(start,0)
		cost_so_far[start] = 0

		while True:
			if frontier.empty():
				return -1
			current = frontier.get()
			if current == self.field.end:
				break
			explored.append(current)

			self.field.set_current(current,"X")

			for child in self.get_diagonal_neighbors(current):

				new_cost = cost_so_far[current] + 1 # You took one step past current

				if child not in explored and self.field.current[child[0]][child[1]] != '*' and child not in cost_so_far.keys():
					if child == self.field.end:
						return 1    #Success
					frontier.put(child, new_cost)
					cost_so_far[child] = new_cost
					self.field.set_current(child,"C")

				if child in cost_so_far.keys():
					if new_cost < cost_so_far[child]:
						cost_so_far[child] = new_cost
						frontier.set_priority(child,new_cost)
						self.field.set_current(child,"R")	#Show when the item is reset
			os.system(CLEAR)
			print("UNIFORM COST SEARCH")
			print()
			print(self.field.to_s_current())
			time.sleep(0.01)

	@method_timing
	def best_first(self):
		'''
			Performs the (greedy) best first search on the current field
		'''
		frontier = PriorityQueue()
		explored = []

		frontier.put(self.field.start,self.heuristic(self.field.start))

		while True:
			if frontier.empty():
				return -1
			current = frontier.get()
			explored.append(current)

			self.field.set_current(current,"X")


			for child in self.get_neighbors(current):

				if child not in explored and self.field.current[child[0]][child[1]] != '*':
					if child == self.field.end:
						return 1    #Success
					frontier.put(child, self.heuristic(child))
					self.field.set_current(child,"C")

			os.system(CLEAR)
			print("BEST-FIRST SEARCH - NO DIAGONALS ALLOWED")
			print()
			print(self.field.to_s_current())
			time.sleep(0.01)

	@method_timing
	def best_first_diagonal(self):
		'''
			Performs the (greedy) best first search on the current field
			and allows diagonal movements
		'''
		frontier = PriorityQueue()
		explored = []

		frontier.put(self.field.start,self.diagonal_distance(self.field.start))

		while True:
			if frontier.empty():
				return -1
			current = frontier.get()
			explored.append(current)

			self.field.set_current(current,"X")


			for child in self.get_diagonal_neighbors(current):

				if child not in explored and self.field.current[child[0]][child[1]] != '*':
					if child == self.field.end:
						return 1    #Success
					frontier.put(child, self.diagonal_distance(child))
					self.field.set_current(child,"C")

			os.system(CLEAR)
			print("BEST-FIRST SEARCH - DIAGONALS ALLOWED")
			print()
			print(self.field.to_s_current())
			time.sleep(0.01)


	@method_timing
	def astar(self):
		steps = 0
		start = self.field.start

		frontier = PriorityQueue()
		frontier.put(start,0)
		came_from = {}
		cost_so_far = {}

		came_from[start] = None
		cost_so_far[start] = 0

		while not frontier.empty():
			current = frontier.get()

			if current == self.field.end:
				break

			self.field.set_current(current,"X")
			os.system(CLEAR)
			print("A* SEARCH - NO DIAGONALS ALLOWED")
			print()

			for child in self.get_neighbors(current):
				print("",child)
				new_cost = cost_so_far[current] + 1 # You took one step past current
				if child not in cost_so_far.keys() or \
				   new_cost < cost_so_far[child]:
					cost_so_far[child] = new_cost
					priority = new_cost + self.heuristic(child)
					frontier.put(child,priority)
					self.field.set_current(child,'C')
					came_from[child] = current

			print(self.field.to_s_current())
			print(current)
			time.sleep(0.05)

		parent = came_from[current]
		while parent != None:
			self.field.set_current(parent,"_")
			parent = came_from[parent]

		os.system(CLEAR)
		print("A* SEARCH - NO DIAGONALS ALLOWED")
		print()
		print(self.field.to_s_current())

	@method_timing
	def astar_diagonal(self):
		steps = 0
		start = self.field.start

		frontier = PriorityQueue()
		frontier.put(start,0)
		came_from = {}
		cost_so_far = {}

		came_from[start] = None
		cost_so_far[start] = 0

		while not frontier.empty():
			current = frontier.get()

			if current == self.field.end:
				break

			self.field.set_current(current,"X")
			os.system(CLEAR)
			print("A* SEARCH - DIAGONALS ALLOWED")
			print()

			'''
				This version shows how A*Star behaves if diagonal movement is
				allowed, and uses the diagonal distance heuristic
			'''
			for child in self.get_diagonal_neighbors(current):
				new_cost = cost_so_far[current] + 1 # You took one step past current
				if child not in cost_so_far.keys() or \
				   new_cost < cost_so_far[child]:
					cost_so_far[child] = new_cost
					priority = new_cost + self.diagonal_distance(child)
					frontier.put(child,priority)
					self.field.set_current(child,'C')
					came_from[child] = current

			print(self.field.to_s_current())
			print(current)
			time.sleep(0.05)

		parent = came_from[current]
		while parent != None:
			self.field.set_current(parent,"_")
			parent = came_from[parent]

		os.system(CLEAR)
		print("A* SEARCH - DIAGONALS ALLOWED")
		print()
		print(self.field.to_s_current())




#=====================
# Main Algorithm

field = Field(20)
field.set_start(0,0)
field.set_end(19,19)
searcher = Searcher(field)
searcher.breadth_first()
input("Press Enter to continue...")
searcher.reset()
searcher.breadth_first_diagonal()
input("Press Enter to continue...")
searcher.reset()
searcher.depth_first()
input("Press Enter to continue...")
searcher.reset()
searcher.depth_first_diagonal()
input("Press Enter to continue...")
searcher.reset()
searcher.best_first()
input("Press Enter to continue...")
searcher.reset()
searcher.best_first_diagonal()
input("Press Enter to continue...")
searcher.reset()
searcher.uniform_cost()
input("Press Enter to continue...")
searcher.reset()
searcher.uniform_cost_diagonal()
input("Press Enter to continue...")
searcher.reset()
searcher.astar()
input("Press a Enter to continue...")
searcher.reset()
searcher.astar_diagonal()
