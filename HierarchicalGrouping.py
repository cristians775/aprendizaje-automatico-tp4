import copy
from random import randint
import numpy as np


class HierarchicalGrouping():
    def __init__(self, data_set, metric) -> None:
        super().__init__()
        self.data_set = copy.deepcopy(data_set)
        self.metric = metric

    def fit(self):

        groups = [[] for i in self.data_set]

        # Asignamos
        for i, ele in enumerate(self.data_set):
            groups[i].append(ele)

        # Comparamos
        for i in range(len(groups)-2):
            k = self.metric.compare(groups[i], groups, i)
            groups[k] = groups[i] + (groups[k])

        return groups


class Centroid:
    def compare(group, group_to_compare):
        print("")


class Min:
    def compare(self, group, group_to_compare, i):

        distances = [distance(group, group_to_compare[j])
                     for j in range(i+1, len(group_to_compare)-1)]
        result_min_distances = [ min(distance) for distance in distances]
        return result_min_distances.index(min(result_min_distances))


class Max:
    def compare(group, group_to_compare):
        print("")


class Prom:
    def compare(group, group_to_compare):
        print("")


def distance(element, element_to_compare):
    distances = [[abs(np.sqrt(sum([pow(x-vector_to_compare[i], 2) for i, x in enumerate(vector)])))
                  for vector in element] for vector_to_compare in element_to_compare]
    return distances
