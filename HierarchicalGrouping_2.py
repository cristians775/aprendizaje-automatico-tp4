import copy
import json
from random import randint
import numpy as np


class HierarchicalGrouping():
    def __init__(self, data_set, metric) -> None:
        super().__init__()
        
        self.metric = metric
        self.data_set = data_set

    def fit(self):
        data_set = [[] for _ in self.data_set]
        data_set_dic = {}
        for i, data in enumerate(self.data_set):
            data_set[i].append(data)
        
        n = len(data_set)
        for i,ele in enumerate(self.data_set):
            data_set_dic[json.dumps(ele, separators=(',', ':'))] = i
       
        for _ in range(0,n-1):
            distances = self.calculate_distances(data_set)
            i, j, dist =  self.metric.compare(distances, data_set)
            print("Grupo: ",self.parse_groups(data_set_dic, data_set[i]),"Merge: ",self.parse_groups(data_set_dic,data_set[j]),"Altura : ", dist )
            data_set[i] = data_set[i] + data_set[j]
            data_set.pop(j)
            n-=1
        print("result", self.parse_groups(data_set_dic, data_set[0]))
    
    def fit_with_avg(self):
        data_set = [[] for _ in self.data_set]
        data_set_dic = {}
        for i, data in enumerate(self.data_set):
            data_set[i].append(data)
        
        n = len(data_set)
        for i,ele in enumerate(self.data_set):
            data_set_dic[json.dumps(ele, separators=(',', ':'))] = i
       
        for _ in range(0,n-1):
            distances = self.calculate_distances(data_set)
            i, j, dist =  self.metric.compare(distances, data_set)
            print("Grupo: ",self.parse_groups(data_set_dic, data_set[i]),"Merge: ",self.parse_groups(data_set_dic,data_set[j]),"Altura : ", dist )
            data_set[i] = data_set[i] + data_set[j]
            data_set.pop(j)
            n-=1
    
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
          
    def parse_groups(self, dic, groups):
        group_parsed = []
        for element in groups:
            group_parsed.append(dic[json.dumps(element, separators=(',', ':'))])
        return group_parsed
class Min():
    def compare(self, distances, data_set=None):
        distances_min = [[ [min(dist) for dist in group] for group in groups]for groups in distances]
        dist_min = None
       
        for i in range(0,len(distances_min)):
            for j in range(0, i):
                if(dist_min == None):
                    dist_min = min(distances_min[i][j])
                    col = i
                    row = j
                elif(min(distances_min[i][j]) < dist_min):
                    dist_min=min(distances_min[i][j])
                    col = i
                    row = j

        return col, row, dist_min
    
    
    
class Max():
    def compare(self, distances, data_set=None):
        distances_max = [[ [max(dist) for dist in group] for group in groups]for groups in distances]
        dist_min = None
        for i in range(0,len(distances_max)):
            for j in range(0, i):
                if(dist_min == None):
                    dist_min = max(distances_max[i][j])
                    col = i
                    row = j
                elif(max(distances_max[i][j]) < dist_min):
                    dist_min=max(distances_max[i][j])
                    col = i
                    row = j
        return col, row, dist_min
    

    
class AVG():
    def compare(self, distances, data_set):
        distances_prom = [[0 for j in distances] for i in distances]
        for i, distance in enumerate(data_set):
            for j, dist in enumerate(data_set):
                print(self.distance(data_set[i], data_set))
                distances_prom[i][j] = (sum([min(distance[i])])+sum([ ele for ele in distance[j]]))/(len(distance[i])+len(distance[j]))
        dist_min = None
        count = 0
        for i in range(0,len(distances_prom)):
            for j in range(0, i):
                if(dist_min == None):
                    dist_min = distances_prom[i][j]
                    col = i
                    row = j
                    count+=1
                elif(distances_prom[i][j] < dist_min):
                    dist_min= distances_prom[i][j]
                    col = i
                    row = j
                    count+=1
    
    

        return col, row, dist_min/count
                
    def distance(self,element, element_to_compare):
        distances = [[abs(np.sqrt(sum([pow(x-vector_to_compare[i], 2) for i, x in enumerate(vector)])))
            for vector in element] for vector_to_compare in element_to_compare]
        return distances 
        