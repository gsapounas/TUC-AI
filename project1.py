#!/usr/bin/python3

import os.path
import sys
import re
import time
import copy
import math
import argparse

from queue import PriorityQueue
from heapq import heappush, heappop

# ida_node_path = []
# ida_rode_path = []
ida_visited_nodes = []

road_matrix = []                            #Holds basic costs and nodes for each road
prediction_road_matrix = []                 #Holds updated costs based on predicted traffic for each road
actual_road_matrix = []                     #Holds updated costs based on actual traffic for each road

prediction_traffic_matrix = []              #Predictions for all 80 days
actual_traffic_matrix = []                  #Actual traffic for all 80 days

p_high = 0.5
p_normal = 0.25
p_low = 0.25

ucs_total_cost_for_80_days = 0
ida_total_cost_for_80_days = 0


############################################################################################################################################################
############################        UTIL FUNCTIONS       ###################################################################################################
############################################################################################################################################################

def count_item(filename, item_name):
    data_copy = False

    file = open(filename, mode = 'r', encoding = 'utf-8-sig')
    items = file.read().splitlines()        #remove newline from line
    file.close()

    count = 0

    for item in items:
        if item.startswith('<' + item_name + '>'):
            data_copy = True
            continue

        if item.startswith('</' + item_name + '>'):
            data_copy = False

        if data_copy:
            count += 1

    return count

def get_source(filename):
    str = ""

    file = open(filename, mode = 'r', encoding = 'utf-8-sig')
    lines = file.read().splitlines()        #remove newline from line
    file.close()

    for line in lines:              #Parse Source, Dest and Roads
        if line.startswith('<Source>'):
            str = re.findall("\>(.*?)\<", line)
            source = ''.join(str)    #convert list to string

            return source

def get_dest(filename):
    str = ""

    file = open(filename, mode = 'r', encoding = 'utf-8-sig')
    lines = file.read().splitlines()        #remove newline from line
    file.close()

    for line in lines:              #Parse Source, Dest and Roads
        if line.startswith('<Destination>'):
            str = re.findall("\>(.*?)\<", line)
            dest = ''.join(str)    #convert list to string

            return dest

def parse_data(filename):
    data_copy1 = False          #Copy flag for outer loop
    data_copy2 = False          #Copy flag for inner loop

    day_prediction_matrix = []
    day_actual_traffic_matrix = []

    file = open(filename, mode = 'r', encoding = 'utf-8-sig')
    lines = file.read().splitlines()        #remove newline from line
    file.close()

    for line in lines:              #Parse Source, Dest and Roads
        if line.startswith('<Roads>'):
            data_copy1 = True
            continue

        if line.startswith('</Roads>'):
            data_copy1 = False

        if data_copy1:
            # print(line)
            road_matrix.append(line.split('; '))

    for line in lines:          #Parse Predictions
        if line.startswith('<Predictions>'):
            data_copy1 = True
            continue

        if line.startswith('</Predictions>'):
            data_copy1 = False

        if data_copy1:
            # print(line)
            if line.startswith('<Day>'):
                data_copy2 = True
                continue

            if line.startswith('</Day>'):
                prediction_traffic_matrix.append(day_prediction_matrix)
                day_prediction_matrix = []
                data_copy2 = False

            if data_copy2:
                day_prediction_matrix.append(line.split('; '))


    for line in lines:          #Parse Actual Traffic Per Day
        if line.startswith('<ActualTrafficPerDay>'):
            data_copy1 = True
            continue

        if line.startswith('</ActualTrafficPerDay>'):
            data_copy1 = False

        if data_copy1:
            # print(line)
            if line.startswith('<Day>'):
                data_copy2 = True
                continue

            if line.startswith('</Day>'):
                actual_traffic_matrix.append(day_actual_traffic_matrix)
                day_actual_traffic_matrix = []
                data_copy2 = False

            if data_copy2:
                day_actual_traffic_matrix.append(line.split('; '))

def create_graph(data_matrix):
    graph = {}

    for i in range(roads):
        # print(data_matrix[i][1], data_matrix[i][2], data_matrix[i][3])
        graph.setdefault(data_matrix[i][1], []).append((data_matrix[i][2], data_matrix[i][3]))
        graph.setdefault(data_matrix[i][2], []).append((data_matrix[i][1], data_matrix[i][3]))

    # print(graph)
    return graph

def update_road_costs(day_idx, road_matrix):                     #Update road costs daily with the predicted traffic
    road_idx = 0

    prediction_road_matrix = copy.deepcopy(road_matrix)
    actual_road_matrix = copy.deepcopy(road_matrix)

    for road in prediction_traffic_matrix[day_idx]:
        if road[1] == 'low':
            if road[0] == road_matrix[road_idx][0]:
                prediction_road_matrix[road_idx][3] = float(road_matrix[road_idx][3]) * 0.9 + p_low

        elif road[1] == 'high':
            if road[0] == road_matrix[road_idx][0]:
                prediction_road_matrix[road_idx][3] = float(road_matrix[road_idx][3]) * 1.25 + p_high

        elif road[1] == 'normal':
            prediction_road_matrix[road_idx][3] = float(road_matrix[road_idx][3]) + p_normal

        road_idx += 1

    road_idx = 0

    for road in actual_traffic_matrix[day_idx]:
        if road[1] == 'low':
            if road[0] == road_matrix[road_idx][0]:
                actual_road_matrix[road_idx][3] = float(road_matrix[road_idx][3]) * 0.9

        elif road[1] == 'high':
            if road[0] == road_matrix[road_idx][0]:
                actual_road_matrix[road_idx][3] = float(road_matrix[road_idx][3]) * 1.25

        elif road[1] == 'normal':
            actual_road_matrix[road_idx][3] = float(road_matrix[road_idx][3])

        road_idx += 1

    return prediction_road_matrix, actual_road_matrix

def calculate_real_cost(day_idx, road_path, actual_road_matrix):
    real_cost = 0

    for road in actual_road_matrix:
        if road[0] in road_path:
            real_cost += float(road[3])

    return real_cost

def find_road_path(graph, node_path):
    road_path = []

    for p in node_path[:-2]:
        q = node_path.index(p)
        location = [r[0] for r in graph[p]].index(node_path[q + 1])
        # predicted_cost = float(graph[p][location][1])

        road_index = find_road(node_path[q], node_path[q + 1])

        if not node_path[q + 1] == dest:
            road_path.append(prediction_road_matrix[road_index][0])
            road_path.append(prediction_road_matrix[road_index][3])


        if node_path[q + 1] == dest:
            road_path.append(prediction_road_matrix[road_index][0])
            road_path.append(prediction_road_matrix[road_index][3])

    return road_path

def find_road(node_A, node_B):
    for index in range(len(prediction_road_matrix)):
        if (node_A == prediction_road_matrix[index][1]):
            if (node_B == prediction_road_matrix[index][2]):

                return index

        elif (node_A == prediction_road_matrix[index][2]):
            if (node_B == prediction_road_matrix[index][1]):

                return index

def remove_duplicate_expensive_roads(graph, prediction_road_matrix):

    # print(graph)
    # print('-------------------------------------------')

    result = {}

    for key, value in graph.items():
        check_val = set()      #Check Flag
        res = []

        for i in value:
            if i[0] not in check_val:
                result.setdefault(key, []).append(i)
                check_val.add(i[0])

    seen = set()
    cond = [x for x in prediction_road_matrix if x[1] not in seen and not seen.add(x[1])]
    # print(cond)
    # print(seen)

    # cheapest_road_matrix = copy.deepcopy([x for x in prediction_road_matrix if x[3] not in seen and not seen.add(x[3])])

    # print(prediction_road_matrix)
    # print('-------------------------------------------')
    # print(cheapest_road_matrix)

    return result


############################################################################################################################################################
########################### Code for redirecting output both to terminal and file ##########################################################################
############################################################################################################################################################
# Code taken from:: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
############################################################################################################################################################

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("Output File.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


############################################################################################################################################################
############################        UCS FUNCTIONS       ####################################################################################################
############################################################################################################################################################
# Code taken and customized from: https://github.com/DeepakKarishetti/Uniform-cost-search/blob/master/find_route.py
############################################################################################################################################################

def ucs(graph, source, destination):
    visited = set()
    ucs_visited_nodes = 0
    node_path = []
    queue = PriorityQueue()
    queue.put((0, [source]))

    while queue:
        # if no path is present beteween two nodes
        if queue.empty():
            print('distance: infinity\nroute: \nnone')
            return

        cost, node_path = queue.get()
        node = node_path[len(node_path) - 1]

        if node not in visited:
            visited.add(node)
            ucs_visited_nodes = ucs_visited_nodes + 1

            if node == destination:
                node_path.append(cost)
                return node_path, ucs_visited_nodes

            for n in neighbors(graph, node):

                if n not in visited:
                    t_cost = cost + float(find_cost(graph, node, n))
                    temp = node_path[:]
                    temp.append(n)
                    queue.put((t_cost, temp))

# function for finding neighbors in the graph
def neighbors(graph, node):
    points = graph[node]
    return [n[0] for n in points]

# function to calculate the cost of path beteween two nodes
def find_cost(graph, node_A, node_B):
    location = [n[0] for n in graph[node_A]].index(node_B)
    cost = graph[node_A][location][1]

    return cost

# output the result of search
def ucs_results_display(ucs_node_path, ucs_road_path, ucs_visited_nodes, day_idx, ucs_real_cost):
    distance = ucs_node_path[-1]

    print('======================================================================================')
    print('Day ', end = '')
    print(day_idx + 1)
    print('UCS:')
    print('    Visited Nodes: ', end = '')
    print(ucs_visited_nodes)

    ucs_exec_time = time.process_time() - start1

    print('    Execution time: ', end = '')
    print(ucs_exec_time, end = ' ')
    print('seconds')
    print('    Path: ', end = '')

    for i in range(0, len(ucs_road_path), 2):
        if not i + 2 == len(ucs_road_path):
            cost =float(ucs_road_path[i + 1])
            cost_rounded = "%.2f" % cost
            print(ucs_road_path[i], '(' + str(cost_rounded) + ')', end = ' -> ')

        else:
            cost =float(ucs_road_path[i + 1])
            cost_rounded = "%.2f" % cost
            print(ucs_road_path[i], '(' + str(cost_rounded) + ')', end = '')

    cost = float(distance)
    cost_rounded = "%.2f" % cost
    print()
    print('    Predicted Cost: ' + str(cost_rounded))

    cost = float(ucs_real_cost)
    cost_rounded = "%.2f" % cost
    print('    Real Cost: ' + str(cost_rounded))


############################################################################################################################################################
############################        IDA FUNCTIONS       ####################################################################################################
############################################################################################################################################################
# Code taken and customized from:
# 1) https://www.algorithms-and-technologies.com/iterative_deepening_a_star/python
# 2) https://en.wikipedia.org/wiki/Iterative_deepening_A*
############################################################################################################################################################

def create_vertex_list(graph):
    keys = graph.keys()
    size = len(keys)

    vertex_list = []

    for key in graph:
        size = len(graph[key])

        for i in range(size):
            if key not in vertex_list:
                vertex_list.append(key)

    return vertex_list


def create_adj_matrix(graph, vertex_list):
    keys = graph.keys()
    size = len(keys)

    adj_matrix = [[0]*size for i in range(size)]

    total_cost = 0
    lowest_cost = 1000
    avg_cost = 0
    count = 0

    for key in graph:
        size = len(graph[key])

        for i in range(size):
            index_a = vertex_list.index(key)
            index_b = vertex_list.index(graph[key][i][0])

            if float(adj_matrix[index_a][index_b]) == 0:                            #Check if cost for the same road is the least, then put it in the matrix
                adj_matrix[index_a][index_b] = graph[key][i][1]

            elif float(graph[key][i][1]) < float(adj_matrix[index_a][index_b]):
                adj_matrix[index_a][index_b] = graph[key][i][1]

            else:
                continue

            if float(graph[key][i][1]) < float(lowest_cost):
                lowest_cost = float(graph[key][i][1])

            total_cost += float(graph[key][i][1])
            count += 1

    avg_cost = total_cost / count

    return adj_matrix

def find_index(vertex_list, value):
    for i in range(len(vertex_list)):
        if vertex_list[i] == value:
            return i

def calculate_heuristic_cost(graph):
    keys = graph.keys()
    size = len(keys)

    heuristic = [[0]*size for i in range(size)]

    for i in range(size):
        for j in range(size):
            if i == j:
                heuristic[i][j] = 0
            else:
                node_path = []
                ucs_visited_nodes = 0
                node_path, ucs_visited_nodes = ucs(graph, vertex_list[i], vertex_list[j])
                # heuristic[i][j] = len(node_path) * avg_cost
                if node_path[-1] != 0:
                    heuristic[i][j] = node_path[-1] - 30 # len(node_path)
                else:
                    heuristic[i][j] = 0

    return heuristic

def iterative_deepening_a_star(adj_matrix, vertex_list, heuristic, start, goal):
    ida_node_path = [start]

    start_index = find_index(vertex_list, start)
    goal_index = find_index(vertex_list, goal)

    threshold = heuristic[start_index][goal_index]

    while True:
        # print("Iteration with threshold: " + str(threshold))

        distance = iterative_deepening_a_star_rec(adj_matrix, vertex_list, heuristic, start, goal, 0, threshold, ida_node_path)

        if distance == math.inf:
            # Node not found and no more nodes to visit
            return -1
        elif distance <= 0:
            # if we found the node, the function returns the negative distance
            # print("Found the node we're looking for!")

            return -distance, ida_node_path
        else:
            # if it hasn't found the node, it returns the (positive) next-bigger threshold
            threshold = distance
            # ida_visited_nodes = []

def iterative_deepening_a_star_rec(adj_matrix, vertex_list, heuristic, node, goal, distance, threshold, ida_node_path):
    # node = ida_node_path[-1]

    # print("Visiting Node " + str(node))

    node_index = find_index(vertex_list, node)
    goal_index = find_index(vertex_list, goal)

    estimate = distance + heuristic[node_index][goal_index]

    if estimate > threshold:
        return estimate

    if node not in ida_visited_nodes:
        ida_visited_nodes.append(node)

    if node == goal:
        return -distance

    # ...then, for all neighboring nodes....
    min = math.inf

    for i in range(len(adj_matrix[node_index])):
        if (adj_matrix[node_index][i] != 0) and (vertex_list[i] not in ida_node_path):

            ida_node_path.append(vertex_list[i])

            t = iterative_deepening_a_star_rec(adj_matrix, vertex_list, heuristic, vertex_list[i], goal, distance + float(adj_matrix[node_index][i]), threshold, ida_node_path)

            if t < 0:
                # Node found
                return t
            elif t < min:
                min = t
                # print(ida_node_path)
            ida_node_path.pop()

    return min

def ida_results_display(ida_node_path, ida_road_path, ida_visited_nodes, day_idx, ida_real_cost):
    distance = ida_node_path[-1]

    print('IDA*:')
    print('    Visited Nodes: ', end = '')
    print(len(ida_visited_nodes))

    ida_exec_time = time.process_time() - start2

    print('    Execution time: ', end = '')
    print(ida_exec_time, end = ' ')
    print('seconds')
    print('    Path: ', end = '')

    for i in range(0, len(ida_road_path), 2):
        if not i + 2 == len(ida_road_path):
            cost = float(ida_road_path[i + 1])
            cost_rounded = "%.2f" % cost
            print(ida_road_path[i], '(' + str(cost_rounded) + ')', end = ' -> ')

        else:
            cost = float(ida_road_path[i + 1])
            cost_rounded = "%.2f" % cost
            print(ida_road_path[i], '(' + str(cost_rounded) + ')', end = '')

    cost = float(distance)
    cost_rounded = "%.2f" % cost
    print()
    print('    Predicted Cost: ' + str(cost_rounded))

    cost = float(ida_real_cost)
    cost_rounded = "%.2f" % cost
    print('    Real Cost: ' + str(cost_rounded))


############################################################################################################################################################
############################        MAIN       #############################################################################################################
############################################################################################################################################################

if __name__ == "__main__":
    print('+------------------------------------------------------------------------------------+')
    print('|                            ╔═══╗  ╔╗╔╗  ╔═╗     ╔╗                                 |')
    print('|                            ║╔═╗║ ╔╝╚╣║  ║╔╝     ║║                                 |')
    print('|                            ║╚═╝╠═╩╗╔╣╚═╦╝╚╦╦═╗╔═╝╠══╦═╗                            |')
    print('|                            ║╔══╣╔╗║║║╔╗╠╗╔╬╣╔╗╣╔╗║║═╣╔╝                            |')
    print('|                            ║║  ║╔╗║╚╣║║║║║║║║║║╚╝║║═╣║                             |')
    print('|                            ╚╝  ╚╝╚╩═╩╝╚╝╚╝╚╩╝╚╩══╩══╩╝                             |')
    print('+------------------------------------------------------------------------------------+')

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_file', type = str, help="This is the path to input file")

    args = parser.parse_args()

    filename = args.input_file
    # filename = sys.argv[-1]     #gets last argument from terminal as filename

    if not os.path.isfile(filename) :
        print('|                              This file does not exist                              |')
        print('+------------------------------------------------------------------------------------+')
        sys.exit(1)

    print('|                                Input file accepted                                 |')
    print('+------------------------------------------------------------------------------------+')
    time.sleep(1)
    print('|                  Please wait, the output file is being generated..                 |')
    print('+------------------------------------------------------------------------------------+')
    time.sleep(1)
    print('|                             Average wait time is 1 min..                           |')
    print('+------------------------------------------------------------------------------------+')

    sys.stdout = Logger()                       # Redirect stdout to file and terminal

############################################################################################################################################################

    roads = count_item(filename, "Roads")
    predictions = count_item(filename, "Predictions")
    actual_traffic = count_item(filename, "ActualTrafficPerDay")

    source = get_source(filename)
    dest = get_dest(filename)

    parse_data(filename)

    day_idx = 0

    prediction_road_matrix = copy.deepcopy(road_matrix)
    actual_road_matrix = copy.deepcopy(road_matrix)

    for day in prediction_traffic_matrix:
        prediction_road_matrix, actual_road_matrix = update_road_costs(day_idx, road_matrix)

        graph = create_graph(prediction_road_matrix)

        vertex_list = create_vertex_list(graph)

        graph = remove_duplicate_expensive_roads(graph, prediction_road_matrix)

        # print(graph)

############################################################################################################################################################
############################        UCS        #############################################################################################################
############################################################################################################################################################

        start1 = time.process_time()

        ucs_node_path = []
        ucs_node_path, ucs_visited_nodes = ucs(graph, source, dest)

        ucs_road_path = []
        ucs_road_path = find_road_path(graph, ucs_node_path)

        ucs_real_cost = calculate_real_cost(day_idx, ucs_road_path, actual_road_matrix)

        if ucs_road_path:
            ucs_results_display(ucs_node_path, ucs_road_path, ucs_visited_nodes, day_idx, ucs_real_cost)

############################################################################################################################################################
############################        IDA*       #############################################################################################################
############################################################################################################################################################

        adj_matrix = create_adj_matrix(graph, vertex_list)

        start2 = time.process_time()

        heuristic = calculate_heuristic_cost(graph)

        cost, ida_node_path = iterative_deepening_a_star(adj_matrix, vertex_list, heuristic, source, dest)

        ida_node_path.append(cost)

        ida_road_path = []
        ida_road_path = find_road_path(graph, ida_node_path)

        ida_real_cost = calculate_real_cost(day_idx, ida_road_path, actual_road_matrix)

        if ida_road_path:
            ida_results_display(ida_node_path, ida_road_path, ida_visited_nodes, day_idx, ida_real_cost)

        # input('Press enter to contiunue...')
        # sys.exit(1)

        # if day_idx == 19:
        #     sys.stdout = sys.__stdout__
        #     print('|                 Do not go anywhere                |')
        #     print('+---------------------------------------------------+')
        #     sys.stdout = open('Output File.txt', 'a')
        #
        # if day_idx == 39:
        #     sys.stdout = sys.__stdout__
        #     print('|                 We are halfway done               |')
        #     print('+---------------------------------------------------+')
        #     sys.stdout = open('Output File.txt', 'a')
        #
        # if day_idx == 59:
        #     sys.stdout = sys.__stdout__
        #     print('|                    Almost there                   |')
        #     print('+---------------------------------------------------+')
        #     sys.stdout = open('Output File.txt', 'a')

        ucs_total_cost_for_80_days += ucs_real_cost
        ida_total_cost_for_80_days += ida_real_cost

        if day_idx == 79:
            avg_ucs_cost = ucs_total_cost_for_80_days / 80
            print('======================================================================================')
            print('Statistics:')
            print('    UCS average real cost for 80 days: ', "%.2f" % avg_ucs_cost)

            avg_ida_cost = ida_total_cost_for_80_days / 80
            print('    IDA* average real cost for 80 days: ', "%.2f" % avg_ida_cost)

        day_idx += 1

    if source not in graph.keys():
        print('Source city not found')
        sys.exit()

    if dest not in graph.keys():
        print('Destination city not found')
        sys.exit()

    sys.stdout = sys.__stdout__

    print('+------------------------------------------------------------------------------------+')
    print('|                                 The file is ready!                                 |')
    print('+------------------------------------------------------------------------------------+')
    print('|                                      Exiting..                                     |')
    print('+------------------------------------------------------------------------------------+')

    time.sleep(1)
