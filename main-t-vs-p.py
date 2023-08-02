
from parallelcolor import *
from statistics import mean
import multiprocessing
from multiprocessing import Pool
from coloredgraph import *
from counter import singleCounter

def drange(start, stop, step):

    i = 0
    while start + step * i <= stop:
        yield start + step * i
        i += 1


def main_work_p_parallel(probability, n_vertices, tries, base1, base2, partitions, step_exponent):
    # generate graphs
    graphs = [partite_graph_probability(n_vertices, partitions, probability) for i in range(tries)]
    average_degree = mean([g.average_degree() for g in graphs])

    stat_simple = singleCounter(['mean', 'std', 'percentage'])
    for g in graphs:
        steps1, steps2, success1, success2 = parallel_color(g=g, parititons=partitions, base1=base1, base2=base2, number_of_phases=2, step_exponent = step_exponent)
     
        stat_simple += [1.0 * (steps1 + steps2),1.0 * (steps1 + steps2), success2 ]
    mean_steps, std_steps, all_success = stat_simple.stats()


    print(partitions, probability, average_degree, n_vertices, tries, step_exponent, mean_steps,  all_success)

    return (partitions, probability, average_degree, n_vertices, tries, step_exponent, mean_steps,  all_success)



if __name__ == '__main__':

    NUMBER_OF_CPUS = multiprocessing.cpu_count()
    print("Using", NUMBER_OF_CPUS, "of", multiprocessing.cpu_count(), "CPUs.")
    pool = Pool(NUMBER_OF_CPUS)

    KONST = list(drange(1.0,8.0,0.1))
    PARTITIONS = [3,4,5,6,7,8,9,10]
    NUM_VERTICES = [60,120,240]

    TRIES = [10000]

    out_ver = "a"

    for num_vertices in NUM_VERTICES:
        for num_part in PARTITIONS:

            STEP_EXPONENT = [
                2.0 #+ 1.5*log(num_part) / log(NUM_VERTICES[0])
                 ]

            print("VERTICES", num_vertices, "PARTITIONS", num_part)

            PROBABILITIES = [1.0 * kon * (num_part - 1.3) / num_vertices for kon in KONST]

            p_parameters = list(it.product(PROBABILITIES,  [num_vertices], TRIES, [4.0], [4.0], [num_part], STEP_EXPONENT))

            print("Number of experiments:", len(p_parameters))

            result = pool.starmap(main_work_p_parallel, p_parameters)

            ff = open(f'pap-{num_vertices}-{num_part}-{out_ver}.csv', 'w')
            print(result)
            for r in result:
                s = ",".join(str(e) for e in r)
                print(s, file=ff)
            ff.close()



