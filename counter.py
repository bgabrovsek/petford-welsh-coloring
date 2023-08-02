import statistics

def data_statistic(data, desc):
    if desc == 'mean' or desc == 'average':
        return statistics.mean(data)
    if desc == 'median':
        return statistics.median(data)
    if desc == 'std' or desc == "stdev" or desc == "sd":
        return statistics.stdev(data)
    if desc == 'min':
        return min(data)
    if desc == 'max':
        return max(data)
    if desc == 'count':
        return sum(bool(x) for x in data)
    if desc == 'percentage' or desc == 'perc':
        return 100.0 * sum(bool(x) for x in data) / len(data)
    if desc == 'sum' or desc == 'add':
        return sum(data)

    raise ValueError("Data description type not implemented")

class multiCounter():

    def __init__(self, levels, value_descriptions):
        self.levels = levels
        self.desc = value_descriptions
        self.values = {key: [] for key in range(self.levels)}


    def __iadd__(self, other):
        if isinstance(other, dict):
            for key in other:
                self.values[key].append(other[key])
        else:
            raise ValueError("Not yet implemented.")
        return self

    def stats(self, level = None):
        result = []
        if level is None:
            for i, d in enumerate(self.desc):
                for l in range(self.levels):
                    n = len(self.values[l])
                    data = [self.values[l][m][i] for m in range(n)]
                    result.append(data_statistic(data, d))

        else:
            raise ValueError("Not yet implemented.")

        return result



class singleCounter():

    def __init__(self, value_descriptions):
        self.desc = value_descriptions
        self.values = []

    def __iadd__(self, other):
        self.values.append(other)
        return self

    def stats(self, level = None):
        return [data_statistic([row[i] for row in self.values], d) for i, d in enumerate(self.desc)]

"""
sc =  singleCounter(['mean', 'percentage', 'std'])

sc += [10.0,True,10.0]
sc += [5,False,5]
sc += [3,True,3]
sc += [10,True,10]

print(sc.stats())
"""
"""
mc = multiCounter(2, ['mean', 'sum', 'percentage', 'count'])
mc += {0:[1,2,1,1], 1:[2,5,6,1]}
mc += {0:[3,1,0,0], 1:[3,0,0,2]}
mc += {0:[9,2,1,1], 1:[2,10,0,1]}
print(mc.stats())
"""
