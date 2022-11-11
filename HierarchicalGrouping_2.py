import copy
from random import randint
import numpy as np


class HierarchicalGrouping():
    def __init__(self, _data_set, metric) -> None:
        super().__init__()
        
        self.metric = metric

    def fit(self):
        data_set = [[[1,2,4,4]], [[5,6,7,8]], [[9,10,11,12]], [[5,2,4,5]]]
                    
        n = len(data_set)
        
        for _ in range(0,n-1):
            distances = self.calculate_distances(data_set)
            i, j, dist =  self.metric.compare(distances)
            print("Fila: ",data_set[i],"Merge: ",data_set[j],"Dist: ", dist )
            data_set[i] = data_set[i] + data_set[j]
            data_set.pop(j)
            n-=1
        print("result", data_set)


    def distance(self,element, element_to_compare):
        
        distances = [[abs(np.sqrt(sum([pow(x-vector_to_compare[i], 2) for i, x in enumerate(vector)])))
            for vector in element] for vector_to_compare in element_to_compare]
        return distances 
        



    def calculate_distances(self, data_set):
        distances = [[0 for j in data_set] for i in data_set]
        for i in range(0,len(data_set)):
            for j in range(0,len(data_set)):
                distances[i][j]=self.distance(data_set[i], data_set[j])
        return distances
          

class Min():
    def compare(self, distances):
        distances_min = [[ [min(dist) for dist in group] for group in groups]for groups in distances]
        dist_min = None
       
        for i in range(0,len(distances_min)):
            for j in range(0, i):
                if(dist_min == None):
                    dist_min = min(distances_min[i][j])
                    col = i
                    row = j
                elif(distances_min[i][j] < dist_min):
                    dist_min=min(distances_min[i][j])
                    col = i
                    row = j

        return col, row, dist_min
    
    
    
class Max():
    def compare(self, distances):
        distances_max = [[ [max(dist) for dist in group] for group in groups]for groups in distances]
        dist_min = None
        for i in range(0,len(distances_max)):
            for j in range(0, i):
                if(dist_min == None):
                    dist_min = min(distances_max[i][j])

                    col = i
                    row = j
                elif(distances_max[i][j] < dist_min):
                    dist_min=min(distances_max[i][j])
                    col = i
                    row = j
        return col, row, dist_min
    

    
class Prom():
    def compare(self, distances, data_set):
        distances_prom = [[0 for j in distances] for i in distances]
        for i, distance in enumerate(distances):
            for j, dist in enumerate(distances):
                distances_prom[i][j] = (sum([sum(ele) for ele in data_set[i]])+sum([ sum(ele) for ele in data_set[j]]))/(len(data_set[i])+len(data_set[j]))
        dist_max = None
        for i in range(0,len(distances)):
            for j in range(0, i):
                if(dist_max == None):
                    dist_max = min(distances_prom[i][j])
                    col = i
                    row = j
                elif(distances_prom[i][j] < dist_max):
                    dist_max=min(distances_prom[i][j])
                    col = i
                    row = j
        return col, row, dist_max
