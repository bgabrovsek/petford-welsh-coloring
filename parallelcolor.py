""""
Petford-Welsh-based algorithm for graph coloring
"""

import itertools as it
import time
from timer import *
from random import random, choices
import numpy as np
from bisection import *
from statistics import mean
import sys
from math import log
from coloredgraph import *



"""def best_Q_phase1(deg):
    QQ6 = {9: 58,
           10: 58, 11: 52, 12: 20, 13: 14, 14: 3.25,
           15: 3.5, 16: 3.5, 17: 4.25, 18: 4.5, 19: 5.5,
           20: 5.5, 21: 6, 22: 7.25, 23: 9.75, 24: 32,
           25: 49, 26: 49.25, 27: 55, 28: 57.5, 29: 43.5}
    if deg not in QQ6: return 64
    return QQ6[deg]

def best_Q_phase2(deg):
    QQ6 = {0:64, 1: 61, 2: 63, 3: 55.5, 4: 7, 5: 6.5,
           6: 5, 7: 4.25, 8: 4, 9: 4.25, 10: 4,
           11: 3.5, 12: 4.75, 13: 4.25, 14: 4.5, 15: 5,
           16: 4.75, 17: 4.75, 18: 6.25, 19: 6, 20: 6,
           21: 64 , 22: 15.75, 23: 32, 24: 32, 25: 15.75,
           26: 20.5, 27: 58, 28: 63.75, 29: 28, 30: 8 ,
           31: 33.0, 32: 63.0, 33: 30.0, 34: 26.0, 35: 24.0}

    #QQ6: {i:4 for i in range(1000)}
    # 36: 55.0, 37:16.25, 38:61.25
    if deg in QQ6:
        return QQ6[deg]
    else:
        return 24
"""

def parallel_color_phase1(g, k, base = None,step_exponent = 1.5):
    """ Parallel Graph Coloring algorithm
    g - graph, k - number of initial colorings, q - weight/temperature
    if no base, best is chosen
    """
    g.colorize(k)  # randomly color the graph with k colors

    assert base is not None

    """if base is None:
        get_base = lambda d_: best_Q_phase1(d_)
    else:
        get_base = lambda d_: base
    """

    # loop through the steps

    for step in range(int(g.n ** step_exponent)):

        old_coloring = list(g.color)  # make a copy of old coloring
        bad_vertices = g.bad_vertices()  # compute bad vertices

        # check if coloring is good
        if len(bad_vertices) == 0:  # is graph properly colored (no bad vertices)
            return step, True

        # change the colors of the vertices (parallel step)
        for v in bad_vertices:
            if random() > .6:
                continue

            nbh_colors = g.neighbourhood_colors(v, coloring = old_coloring)  # all colors in neighbourhood

            g.color[v] = choices(population=g.list_of_colors,
                                   weights=[base ** (-nbh_colors[color]) for color in g.list_of_colors],
                                   k=1)[0]

    return int(g.n ** step_exponent), False



def parallel_color_phase1_variable_base(g, k, var_base = None, step_exponent = 1.5):
    """ Parallel Graph Coloring algorithm
    g - graph, k - number of initial colorings, q - weight/temperature
    if no base, best is chosen
    """
    g.colorize(k)  # randomly color the graph with k colors

    assert var_base is not None

    """if base is None:
        get_base = lambda d_: best_Q_phase1(d_)
    else:
        get_base = lambda d_: base
    """

    # loop through the steps

    for step in range( int(g.n ** step_exponent)):

        old_coloring = list(g.color)  # make a copy of old coloring
        bad_vertices = g.bad_vertices()  # compute bad vertices

        # check if coloring is good
        if len(bad_vertices) == 0:  # is graph properly colored (no bad vertices)
            return step, True

        # change the colors of the vertices (parallel step)
        for v in bad_vertices:
            if random() > .6:
                continue

            nbh_colors = g.neighbourhood_colors(v, coloring = old_coloring)  # all colors in neighbourhood

            g.color[v] = choices(population=g.list_of_colors,
                                   weights=[var_base[g.degree(v)] ** (-nbh_colors[color]) for color in g.list_of_colors],
                                   k=1)[0]

    return int(g.n ** step_exponent), False


def parallel_color_phase2(g, k, base, step_exponent):
    """ parallel graph coloring algorithm, g - graph, k - number of initial colorings, q - weight """

    assert base is not None

    # split vertices into partitions based on previous coloring
    partitions = g.partitions()

    g.colorize(k)  # recolor the graph with fewer colors

    bad_vert_count_v2 = len(g.bad_vertices())  # count bad vertices

    for step in range(int(g.n ** step_exponent)):

        if bad_vert_count_v2 == 0:
            return step, True

        old_c = list(g.color)  # make a copy
        bad_vertices = g.bad_vertices(partitions[step % (k * 2)]) #{v for v in partitions[step % prev_k] if g.bad_vertex(v)}

        bv1 = len(g.bad_vertices())

        # change the vertices (parallel step)
        for v in bad_vertices:

            local_bad_vert_count = len(g.bad_vertices([v] + g.neighbours[v]))

            cnc = g.neighbourhood_colors(v, coloring=old_c)

            g.color[v] = choices(population=g.list_of_colors,
                           weights=[base ** (-cnc[color]) for color in g.list_of_colors],
                                 k=1)[0]

            bad_vert_count_v2 = bad_vert_count_v2 - local_bad_vert_count + len(g.bad_vertices([v] + g.neighbours[v]))  # count bad vertices

    return int(g.n ** step_exponent), False



def parallel_color_phase2_variable_base(g, k, var_base, step_exponent):
    """ parallel graph coloring algorithm, g - graph, k - number of initial colorings, q - weight """

    assert var_base is not None

    # split vertices into partitions based on previous coloring
    partitions = g.partitions()

    g.colorize(k)  # recolor the graph with fewer colors

    bad_vert_count_v2 = len(g.bad_vertices())  # count bad vertices

    for step in range(int(g.n ** step_exponent)):

        if bad_vert_count_v2 == 0:
            return step, True

        old_c = list(g.color)  # make a copy
        bad_vertices = g.bad_vertices(partitions[step % (k * 2)]) #{v for v in partitions[step % prev_k] if g.bad_vertex(v)}

        bv1 = len(g.bad_vertices())

        # change the vertices (parallel step)
        for v in bad_vertices:

            local_bad_vert_count = len(g.bad_vertices([v] + g.neighbours[v]))

            cnc = g.neighbourhood_colors(v, coloring=old_c)

            g.color[v] = choices(population=g.list_of_colors,
                           weights=[var_base[g.degree(v)] ** (-cnc[color]) for color in g.list_of_colors],
                                 k=1)[0]

            bad_vert_count_v2 = bad_vert_count_v2 - local_bad_vert_count + len(g.bad_vertices([v] + g.neighbours[v]))  # count bad vertices

    return int(g.n ** step_exponent), False


def parallel_color(g, parititons, base1, base2, number_of_phases = 2, step_exponent = 1.5):

    if number_of_phases not in [1,2]:
        raise ValueError("number of phases should be 1 or 2.")

    step1, suc1 = parallel_color_phase1(g, parititons*2, base=base1, step_exponent = step_exponent)

    if number_of_phases == 1:
       return step1, 0, suc1, False

    if suc1:
        step2, suc2 = parallel_color_phase2(g, parititons, base=base2, step_exponent = step_exponent)
    else:
        step2, suc2 = int(g.n ** step_exponent), False

    return step1, step2, suc1, suc2


    
def parallel_color_variable_base(g, parititons, var_base, number_of_phases = 2, step_exponent= 1.5):

    if number_of_phases not in [1,2]:
        raise ValueError("number of phases should be 1 or 2.")

    step1, suc1 = parallel_color_phase1_variable_base(g, parititons*2, var_base=var_base, step_exponent= step_exponent)

    if number_of_phases == 1:
       return step1, 0, suc1, False

    if suc1:
        step2, suc2 = parallel_color_phase2_variable_base(g, parititons, var_base=var_base, step_exponent= step_exponent)
    else:
        step2, suc2 = int(g.n ** step_exponent), False

    return step1, step2, suc1, suc2