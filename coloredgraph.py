from math import floor
from random import choices, random, shuffle, choice, sample
import itertools as it
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def str_array_abbr(a, n = 6):
    return " ".join(str(x_) for x_ in a[:min(n, len(a))]) + (" ..." if len(a) > n else "")

class ColoredGraph:
    """ Simple unoriented colored graph  """

    def __init__(self, n):
        self.n, self.e = n, 0  # number of vertices/edges
        self.edges = []  # list of edges (tuples [a,b], a < b)
        self.vertices = list(range(self.n))
        self.neighbours = [[] for v in range(n)]  # vertex adjacency list
        self.list_of_colors = []  # list of available colors
        self.color = [None for v in range(n)]  # list of colors

    def add_edges(self, edge_list):
        """ add edges to graph """
        edge_list = sorted([tuple(sorted(e)) for e in edge_list])  # sort edges
        self.edges += edge_list
        for e in edge_list:
            self.neighbours[e[0]].append(e[1])
            self.neighbours[e[1]].append(e[0])
        self.e += len(self.edges)

    def append_edge(self, edge):
        """ add edges to graph """
        if edge[0] < edge[1]:
            self.edges.append((edge[0], edge[1]))
        else:
            self.edges.append((edge[1],edge[0]))
        self.neighbours[edge[0]].append(edge[1])
        self.neighbours[edge[1]].append(edge[0])
        self.e += 1

    def remove_edge(self, edge):
        edge2 = tuple(sorted(edge))



        self.edges.remove(edge2)
        self.neighbours[edge2[0]].remove(edge2[1])
        self.neighbours[edge2[1]].remove(edge2[0])


    def colorize(self, k):
        """ color by k colors randomly """
        self.list_of_colors = list(range(k))  # list of all possible colors/partitions
        self.color = choices(self.list_of_colors, k=self.n)

    def good_coloring(self):
        for a, b in self.edges:
            if self.color[a] == self.color[b]:
                return False

        return True


    def bad_vertex(self, vertex: int) -> bool:
        """ return true if v is a bad vertex"""
        return any(self.color[w] == self.color[vertex] for w in self.neighbours[vertex])

    def bad_vertices(self, verts=None, coloring=None):
        """ returns set of vertices that are badly colored, optionally: just a subset of verts or alternative colors"""
        if coloring is None: coloring = self.color
        if verts is None:
            return {v for e in self.edges if coloring[e[0]] == coloring[e[1]] for v in e}
        else:
            return {v for v in verts if self.bad_vertex(v)}

    def degree(self, v):
        return len(self.neighbours[v])

    def degrees(self):
        return [self.degree(v) for v in self.vertices]

    def average_degree(self):
        return np.mean([self.degree(v) for v in self.vertices])

    def neighbourhood_colors(self, v, coloring=None):
        """ returns colors in neighbourhood as a dictionary """
        if coloring is None:
            coloring = self.color
        color_count_dict = {c: 0 for c in self.list_of_colors}
        for w in self.neighbours[v]:
            color_count_dict[coloring[w]] += 1
        return color_count_dict

    def vertices_of_color(self, color):
        """ returns vertices with color """
        return [v for v in range(self.n) if self.color[v] == color]

    def partitions(self):
        """ split vertices into partitions """
        return [self.vertices_of_color(color) for color in range(len(self.list_of_colors))]

    def stats(self):
        """ prints some stats about the graph """
        print("Vertices:", self.n)
        print("Edges:", self.e)
        dgs = [self.degree(v) for v in self.vertices]
        print("Average degree:", np.mean(dgs), "min", np.min(dgs), "max", np.max(dgs))
        print("Duplicate edges:", np.sum([v for v in Counter(self.edges).values() if v > 1]))
        print("Isolated vertices:", np.sum([1 for v in dgs if v == 0]))

        #print(self.neighbours)
        #print(self.edges)

    def simpleQ(self):
        # no loops
        for e in self.edges:
            if len(set(e)) != 2:
                return False
        # no duplicate edges
        return  len(self.edges) == len({set(e) for e  in self.edges})


    def regularQ(self):
        # all vertices are equal degree
        deg_count = [0 for i in range(self.n)]
        for e in self.edges:
            deg_count[e[0]] += 1
            deg_count[e[1]] += 1
        return all(deg_count[0] == deg_count[i] for i in range(self.n))

    def partiteQ(self, k):
        pass

    def __repr__(self):
        """ string representation """
        return "Graph ({}/{}) ".format(self.n, self.e) + str_array_abbr(self.edges) \
               + " [" + str_array_abbr(self.color) + "]"

def partite_graph_probability(n, k, p):
    """ Generates k-partite graph with n vertices and p probability of edges (2nd version) """
    g = ColoredGraph(n)
    c = [floor(i * k / n) for i in range(n)]  # partition colors
    g.add_edges([h for h in it.combinations(list(range(n)), 2) if c[h[0]] != c[h[1]] and random() < p])
    return g

def split_range(n, k):
    """ splits range(0,n) into k partitions of almost equal size, diff is max 1"""
    partitions, v = [], 0
    for i in range(k):
        partitions.append((v, v + n//k + (int)(i < n % k)))
        v += n//k + (int)(i < n % k)
    return partitions

def multirange(ranges):
    """ (0,3), (5,8) -> (0,1,2,5,6,7) """
    result = []
    for r in ranges:
        result += list(range(*r))
    return result

def normal_dist(mean, scale, size, minimum = None, maximum = None):
    """ get randomly distributed integers, forced between min and max"""
    # TODO: make faster
    nums = np.zeros(size)
    for i in range(size):
        while True:
            nums[i] = round(np.random.normal(loc=mean, scale=scale, size=None))
            if not (((minimum is not None) and (nums[i] < minimum)) or ((maximum is not None) and (nums[i] > maximum))):
                break
    return nums

def partite_graph_degree(n, k, deg):
    """ Generates k-partite graph (on average) deg-degree """
    g = ColoredGraph(n)

    vert_part = [floor(i * k / n) for i in range(n)]  # to which partition does a vertex belong?
    part_range = split_range(n, k)  # ranges of partitions
    degrees = np.array(normal_dist(mean=deg, scale=0.75, size=n, minimum=1, maximum=deg*2-1), dtype=int)  # generate random degrees
    #deg_dict = dict(enumerate(degrees))  # degrees to dict
    adj_part_verts = {ind: np.array(multirange(part_range[:ind] + part_range[ind+1:])) for ind in range(k)}  # vertices of adjacent partitions
    adj_part_degs = {ind: degrees[adj_part_verts[ind]] for ind in range(k)}  # degrees of vertices in partitions
    avail_verts = list(range(n))

    where_vert = {v: [] for v in range(n)}  # where are certain vertices located in adj_part_verts, so we cna change the degree
    for p in range(k):
        for i, v in enumerate(adj_part_verts[p]):
            where_vert[v].append((p,i))

    """
    print("vert part", vert_part)
    print("degrees", degrees)
    print("adj part", adj_part_verts)
    print("adj part", adj_part_degs)
    print("avail vert", avail_verts)
    print("where", where_vert)
    """

    while bool(avail_verts):
        v = choice(avail_verts)
        v_part = vert_part[v]
        num_choices = sum(adj_part_degs[v_part])
        if num_choices > 0:
            w = np.random.choice(adj_part_verts[v_part], p=adj_part_degs[v_part] / num_choices)
            #print("vert", v, "->", w, " (choices ", num_choices, ")")
            g.append_edge((v,w))

            for u in (v,w):
                degrees[u] -= 1
                if degrees[u] <= 0:
                    avail_verts.remove(u)
                    #print("remove", u)
                for p,i in where_vert[u]:
                    adj_part_degs[p][i] = max(adj_part_degs[p][i]-1, 0)

        else:
            avail_verts.remove(v)
            pass

    # remove duplicate edges and randomly reconnect
    duplicates = dict(Counter(g.edges))
    duplicates = {edge: duplicates[edge] for edge in duplicates if duplicates[edge] > 1}
    #print("duplicates:", duplicates, g.edges)
    while bool(duplicates):
        e = choice(list(duplicates))
        g.remove_edge(e)
        duplicates[e] -= 1
        if duplicates[e] <= 1:
            del duplicates[e]

        v, w = e if choice([True, False]) else (e[1], e[0])
        # try to make new connection
        candidates = set(adj_part_verts[vert_part[v]]) - set(g.neighbours[v])  # all possible candidates
        if len(candidates) > 0:
            w2 = choice(list(candidates))
            g.append_edge((v,w2))
        # TODO: if v fails, then try w (on big enough graphs, this shouldn't happen

    # add isolated vertices

    # add additional edges


    g.color = list(vert_part)
    #rint(g)
    #print("finish")
    #print(g)
    #g.stats()
    #exit()

    return g


    #print(next(iter(deg_dict)))


    for partition_index, rng in enumerate(partitions):
        # possible vertices to connect
        vertices = np.array(multirange(partitions[:partition_index] + partitions[partition_index+1:]))  # get list of vertices not in p_ind


        for v in range(*rng): # loop through all vertices in partitions
            for missing_edge_index in range(degrees[v]):
                vert_deg = degrees[vertices]  # degrees to connect the vertices

                if np.sum(vert_deg) > 0:
                    probabilities = vert_deg / np.sum(vert_deg)
                    w = np.random.choice(vertices, p=probabilities)
                    degrees

            #g.add_edges()
            pass
        print()

    return g

class degdict:
    def __init__(self, max_deg, n):
        self.setdict = {i: set() for i in range(0,max_deg)}  # all are empty
        self.setdict[max_deg] = set(range(n))  # except max is full
        self.indices = [max_deg for i in range(n)]  # indices/degrees of verts

    def available(self, ss = None):  # are there any available degrees?
        if ss is None:
            return any(len(self.setdict[i]) for i in self.setdict if i)
        else:
            return any(len(self.setdict[i] & ss) for i in self.setdict if i)

    def max_set(self, ss = None):  # return the maximal set, if available
        for i in range(max(self.setdict),0,-1):
            sd = self.setdict[i] if ss is None else (self.setdict[i] & ss)
            if len(sd):
                return sd  # return set of verts and index/deg

        raise ValueError("cannot access empty degdict.")

    def random_vertex(self, ss = None):  # return random vertex with maximal degree
        return choice(tuple(self.max_set(ss)))  # TODO: speed up

    def decrease(self, elt):
        i_ = self.indices[elt]
        self.setdict[i_].remove(elt)
        self.setdict[i_-1].add(elt)
        self.indices[elt] = i_-1

def shuffled_range(n):
    x = list(range(n))
    shuffle(x)
    return x

def _exact_partite_graph_degree_(n, k, deg):
    """ Generates k-partite graph with exactly deg-degree """
    if n % k != 0:
        raise ValueError("n must be multiple of k.")

    P = n // k  # partition size

    g = ColoredGraph(n)
    dd = degdict(deg, n) # degree dictionary
    adj_part_verts = {p: set(v for v in range(n) if v // P != p) for p in range(k)}  # vertices of adjacent partitions, to which we can connect

    # main loop
    for v in shuffled_range(n):
        for l in range(dd.indices[v]):  # missing degrees
            non_adj = adj_part_verts[v//P] - set(g.neighbours[v])
            #print(g.edges)
            if dd.available(non_adj):
                w = dd.random_vertex(non_adj)
                g.append_edge((v,w))
                dd.decrease(v)
                dd.decrease(w)
            else:
                #print(dd.indices[v], g.edges)
                raise ValueError("no edges left for vertex " + str(v))

    return g

def regular_graph(g):
    if len(set(g.edges)) != len(g.edges): return False
    if any(g.degree(v) != g.degree(0) for v in g.vertices): return False
    return True

def fast_exact_partite_graph(n, k, deg):
    """ Steger/Wormald
    for n = 6, k = 2, deg = 3 we have 6 groups:
    [1 2 3 | 4 5 6 | 7 8 9 ] [10 11 12 | 13 14 15 | 16 17 18]
    """
    m = (n//k)  # partition size
    deg_m = deg * m
    counter = 0
    while True:
        if counter > 50:
            raise ValueError("Cannot generate graph in 50 tries.")
        counter += 1
        unpaired_points = list(range(n * deg))  # set U, unpaired points
        shuffle(unpaired_points)
        #print(unpaired_points)
        #groups =  {g: {i for i in range(n) if (g // m) != (i // m)} for g in range(n)}  # groups available to other groups
        #print("U =", unpaired_points)
        adjacent = [set() for i in range(n)]  # each vertex has a list of adjacent vertices
        while unpaired_points:
            point0 = unpaired_points.pop()  # take the last unmatched point
            #print(point0, point0//deg, end=" -> ")

            point1 = None

            for _point1 in unpaired_points:
                # points must not be in the same parition
                if _point1 // deg_m == point0 // deg_m:
                    continue
                # the edge must be a new one
                if (_point1//deg) in adjacent[point0//deg]:
                    continue

                point1 = _point1

            if point1 is None:
                break

            #print(point1, point1//deg)
            adjacent[point0//deg].add(point1//deg)
            adjacent[point1//deg].add(point0//deg)
            unpaired_points.remove(point1)

        if not unpaired_points:
            return adjacent

    return None

#print(fast_exact_partite_graph(6, 2, 3))
#print(fast_exact_partite_graph(90, 3, 3))
#fast_exact_partite_graph(6, 3, 2)

def exact_partite_graph_degree(n, k, deg):
    """ tries generating graph until no fails. TODO: make faster w/o retries"""
    if n % k != 0:
        raise ValueError("n must be multiple of k.")

    g = None
    while True:
        try:
            g = _exact_partite_graph_degree_(n, k, deg)
            break
        except:
            pass
    if not regular_graph(g):
        raise ValueError("generated grah not regular")

    return g


#g = exact_partite_graph_degree(120, 3, 5)
#print(g)

#g = partite_graph_degree(300, 3, 3)
#print(g)
#print(g.average_degree())
#print(g.e)
#exit()

# TEST
"""
g = ColoredGraph(4)
g.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
g.colorize(2)
g.color = [0,0,0,1]
print("g", g.n, g.edges, g.color)
print(g.bad_vertices())
print(g.bad_vertices(verts=[0,1, 3]))
print(g.bad_vertices(verts=[0,1, 3], coloring=[0,1,1,1]))
print(g.neighbourhood_colors(1))
print(g.neighbourhood_colors(1, coloring=[0,1,1,1]))
print(g.vertices_of_color(0))
print(g.partitions())
print(g)
"""




