from cmath import e
import random
from tkinter import W
import numpy as np

class Kohonen():
    def __init__(self, data_set, y , k) -> None:
        super().__init__()
        self.k=k
        self.data_set = data_set
        self.labels = [[[] for j in range(0, self.k)] for i in range(0, self.k)]
        self.y = y
    def fit(self):
        
        weights = [[[] for j in range(0, self.k)] for i in range(0, self.k)]
        for i in range(0, self.k):
            for j in range(0, self.k):
                weights[i][j]= self.data_set[random.randint(0, len(self.data_set)-1)]
                self.labels[i][j] = {'enfermo':0, 'sano':0}
        data_set_random = [[] for i in range(0,len(self.data_set))]
        
        for i in range(0, len(self.data_set)):
                data_set_random[i]= self.data_set[random.randint(0, len(self.data_set)-1)]
        
        vicinity = self.k
        etha = 0.1
        times = 0
        max_times = 200*len(data_set_random[0])
        
        while(times< max_times):
            for i in range(0,len(data_set_random)-1):
                    distances = self.calculate_distances(data_set_random[i], weights)
                    i_min = distances.index(min(distances))
                    j_min = distances[i_min].index(min(distances[i_min]))
                    weight_min = weights[i_min][j_min]
                    delta_w = [ etha * (data_set_random[i][k]-weight) for k,weight in enumerate(weight_min) ]
                    weights[i_min][j_min] = [ weight + delta_w[k] for k,weight in enumerate(weight_min)]
                    if(self.y[i]==0): 
                        self.labels[i_min][j_min]['sano']+=1
                    elif(self.y[i]==1):
                         self.labels[i_min][j_min]['enfermo']+=1
                    distances_v = self.calculate_distances(weight_min, weights)
                    
                    for col in range(0, len(distances_v)-1):
                        for row in range(0, len(distances_v)-1):
                            if(distances_v[col][row]< vicinity and col!=i_min and row!=j_min):
                                V = e**((-2*distances_v[col][row])/vicinity)
                                delta_w = [V*etha*(data_set_random[i][k]-weight) for k,weight in enumerate(weights[col][row])]
                                weights[col][row] = [ weight+delta_w[k] for k,weight in enumerate(weights[col][row])]
            etha = 0.1*(1-times/max_times)
            times+=1
            print( times)
            vicinity=((max_times-times)*self.k)/max_times
    
        return self.labels, weights
  
   
    def calculate_distances(self, weight, weights):
        distances = [[[] for j in weights] for i in weights]
        for i in range(0,len(weights)):
            for j in range(0,len(weights)):
                distances[i][j] = abs(np.sqrt(sum([pow(weight[k]-x, 2) for k, x in enumerate(weights[i][j])])))  
        return distances
    
  