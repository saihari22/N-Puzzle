from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
from collections import deque
import psutil
import heapq

#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []
        self.total_cost = calculate_total_cost(self)

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])


    def __lt__(self, child):
            if self.total_cost != child.total_cost:
                if(self.total_cost < child.total_cost):
                    return True
                else:
                    return False
            else:
                actionOrder = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3, 'Initial': -1}
                if (actionOrder[self.action] < actionOrder[child.action]):
                    return True
                else:
                    return False

    def swapTile(self, left, right):
        tempState = self.config[:]
        temp=tempState[left]
        tempState[left]=tempState[right]
        tempState[right]=temp
        return tempState

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index >= 3:
            tempConfig=self.swapTile(self.blank_index, self.blank_index - 3)
            return PuzzleState(tempConfig, self.n, self, 'Up', self.cost + 1)
        else:
            return None
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index <= 5:
            tempConfig = self.swapTile(self.blank_index, self.blank_index + 3)
            return PuzzleState(tempConfig, self.n, self, 'Down', self.cost + 1)
        else:
            return None

    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index != 0 and self.blank_index != 3 and self.blank_index != 6:
            tempConfig = self.swapTile(self.blank_index, self.blank_index - 1)
            return PuzzleState(tempConfig, self.n, self, 'Left', self.cost + 1)
        else:
            return None

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index != 2 and self.blank_index != 5 and self.blank_index != 8:
            tempConfig = self.swapTile(self.blank_index, self.blank_index + 1)
            return PuzzleState(tempConfig, self.n, self, 'Right', self.cost + 1)
        else:
            return None

    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput():
    ### Student Code Goes here
    process = psutil.Process()
    max_ram_usage = (process.memory_full_info().peak_wset) * (10 ** -6)
    file = open(f'output.txt', 'w')
    file.write('path_to_goal: ' + str(output["path"]) + '\n')
    file.write('cost_of_path: ' + str(len(output["path"])) + '\n')
    file.write('nodes_expanded: ' + str(output["nodes_expanded"]) + '\n')
    file.write('search_depth: ' + str(output["search_depth"]) + '\n')
    file.write('max_search_depth: ' + str(output["max_search_depth"]) + '\n')
    file.write('running_time: ' + str(round(total_time,8))+ '\n')
    file.write('max_ram_usage: ' + str(round(max_ram_usage,8))+ '\n')
    file.close()

def findPath(finalState):
    tempState = finalState
    pathList = [tempState]
    while tempState.parent is not None:
        pathList.append(tempState.parent)
        tempState = tempState.parent
    return pathList

def path(node):
    path=[]
    for node in findPath(node)[-2::-1]:
        path.append(node.action)
    return path

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    frontier = deque()
    explored = set()
    finalState = None
    nodes_expanded = 0
    max_depth = 0

    frontier.append(initial_state)
    global output
    output = {}

    while frontier is not None:
        PuzzleState = frontier.popleft()
        explored.add(tuple(PuzzleState.config))
        if test_goal(PuzzleState):
            finalState = PuzzleState
            nodes_expanded = len(explored) - len(frontier) - 1
            break
        PuzzleState.expand()
        for neighbor in PuzzleState.children:
            if tuple(neighbor.config) not in explored:
                frontier.append(neighbor)
                explored.add(tuple(neighbor.config))
                max_depth = max(max_depth, neighbor.cost)
    path1 = path(finalState)
    output = {"path": path1, "search_depth": finalState.cost, "nodes_expanded": nodes_expanded,
              "max_search_depth": max_depth}


def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    frontier = deque()
    explored = set()
    finalState = None
    nodes_expanded = 0
    max_depth = 0
    frontier.append(initial_state)
    global output
    output = {}
    while frontier:
        PuzzleState = frontier.pop()
        explored.add(tuple(PuzzleState.config))
        if test_goal(PuzzleState):
            finalState = PuzzleState
            nodes_expanded = len(explored) - len(frontier) - 1
            break
        PuzzleState.expand()
        for neighbor in PuzzleState.children[::-1]:
            if tuple(neighbor.config) not in explored:
                frontier.append(neighbor)
                explored.add(tuple(neighbor.config))
                max_depth = max(max_depth, neighbor.cost)
    path1 = path(finalState)
    output = {"path": path1, "search_depth": finalState.cost, "nodes_expanded": nodes_expanded,
              "max_search_depth": max_depth}


def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    frontier = []
    explored = set()
    finalState = None
    nodes_expanded = 0
    max_depth = 0
    frontier.append(initial_state)
    global output
    output = {}
    heapq.heappush(frontier, initial_state)
    while frontier is not None:
        PuzzleState = heapq.heappop(frontier)
        explored.add(tuple(PuzzleState.config))
        if test_goal(PuzzleState):
            finalState = PuzzleState
            nodes_expanded = len(explored) - len(frontier) - 1
            break
        PuzzleState.expand()
        for neighbor in PuzzleState.children[::-1]:
            if tuple(neighbor.config) not in explored:
                heapq.heappush(frontier, neighbor)
                explored.add(tuple(neighbor.config))
                max_depth = max(max_depth, neighbor.cost)

    path1 = path(finalState)
    output = {"path": path1, "search_depth": finalState.cost, "nodes_expanded": nodes_expanded,
              "max_search_depth": max_depth}


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    sum = 0
    for i, val in enumerate(state.config):
        if val != 0:
            sum = sum + calculate_manhattan_dist(i, val, state.n)
    return sum + state.cost

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    goal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    goal_index = goal.index(value)
    y_dist = abs(goal_index // n - idx // n)
    x_dist = abs(goal_index % n - idx % n)
    return y_dist + x_dist

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    if puzzle_state.config == goal_state:
        return True
    else:
        return False

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    global total_time
    total_time = end_time - start_time
    print("Program completed in %.3f second(s)"%(end_time-start_time))
    writeOutput()

if __name__ == '__main__':
    main()
