
import copy
from json.encoder import INFINITY
import math
import random
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from HierarchicalGrouping_2 import AVG, HierarchicalGrouping, Max, Min


from KMeans import KMeans
from Kohonen import Kohonen

df = pd.read_excel("./data.xlsx")

# Conjunto de datos
data_set = df.values.tolist()

# Filtramos los datos y quitamos las filas que tienen valores de los atributos en blanco
data_set = list(filter(lambda element: not np.isnan(element[3]), data_set))

# Atributos segun su posicion
attibutes_dic = {
    'sex': 0,
    'age': 1,
    'cad.dur': 2,
    'choleste': 3,
    'sigdz': 4,
    'tvdlm': 5
}
# Estandarizamos los atributos

def standardize_attributes(data, attribute_names):
    # Obtenemos el tamaño del conjunto
    length = len(data)
    # Recorro los elementos seleccionados [ 'age', 'cad.dur', 'choleste' ] -> obtengo la media, el desvio y el valor standar 
    # segun la columna en la que estamos parados
    for name in attribute_names:
        atributte_position = attibutes_dic[name]
        # obtengo la media de los atributos elegidos
        prom = sum(map(lambda element: element[atributte_position], data)) / length
        # Obtengo la desviacion
        # Pow -> potencia de un numero -> pow(2,3) -> 8
        # math.sqrt -> raiz cuadrada
        standard_deviation = math.sqrt(sum(map(lambda element: pow((element[atributte_position]-prom), 2), data))/(length - 1))
        standard_variable = list(map(lambda element: (element[atributte_position]-prom)/standard_deviation,data))
        # Recorro el array original y le reemplazo los nuevos nuevos valores al atributo correspondiente
        for i in range(len(data)):
            data[i][atributte_position] = round(standard_variable[i],3)
# Descartamos stdgiz
def calculate_accuracy(true_class, result_class):
    print("length true class: ", len(true_class), " length result class: ", len(result_class))
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(0, len(result_class)):
        if (true_class[i] == 1 and result_class[i] == 1):
            tp += 1
        elif (true_class[i] == 0 and result_class[i] == 0):
            tn += 1
        elif ((true_class[i] == 0 and result_class[i] == 1)):
            fp += 1
        elif ((true_class[i] == 1 and result_class[i] == 0)):
            fn += 1
    return (tp+tn)/(tp+tn+fp+fn)

                     
def get_data_random(data, n):
    data_random = []
    for _ in range(0,n):
        data_random.append(data[random.randint(0,len(data)-1)])    
    return data_random

standardize_attributes(data_set,[ 'age', 'cad.dur', 'choleste' ])

def kmeans(k, data_set):
    data = copy.deepcopy(data_set)
    sigdz_list = []
    sigdz_position = attibutes_dic['sigdz']

    new_data = [[] for i in data]
    for i, vector in enumerate(data):
        # No tomamos en cuenta el atributo tvdlm
        sigdz_list.append(vector[sigdz_position])
        new_data[i] = vector[:len(vector)-2]
    
    kmeans = KMeans(k, new_data)
    kmeans_result = kmeans.fit()
    k_attribute_position = 4
    
    
    accuracy = calculate_accuracy(sigdz_list, list(map(lambda x: x[k_attribute_position],kmeans_result)))
    return accuracy



#print("KMEANS ACCURACY RESULTADO CON K = 2 CON 750 DATOS DEL CONJUNTO", kmeans(2, get_data_random(data_set, 750)))
#print("KMEANS ACCURACY RESULTADO CON K = 2 CON 1000 DATOS DEL CONJUNTO", kmeans(2, get_data_random(data_set, 1000)))
""" print("KMEANS ACCURACY RESULTADO CON K = 3", kmeans(3, data_set))
print("KMEANS ACCURACY RESULTADO CON K = 4", kmeans(4, data_set))
print("KMEANS ACCURACY RESULTADO CON K = 5", kmeans(5, data_set))
print("KMEANS ACCURACY RESULTADO CON K = 6", kmeans(6, data_set))
print("KMEANS ACCURACY RESULTADO CON K = 7", kmeans(7, data_set))
print("KMEANS ACCURACY RESULTADO CON K = 8", kmeans(8, data_set)) """
def plot_dendogram(data_set, metric):
    print('------------------------',metric )  
    dendrogram = sch.dendrogram(sch.linkage(data_set, method=metric))
    
    plt.title('Dendograma')
    plt.xlabel('Indice - Fila')
    plt.ylabel('Distancia')
    plt.show()

        
def hg(data, metric_name,n):
    data_random = get_data_random(data,n)
    if(metric_name=='single'):
        hg = HierarchicalGrouping(data_random, Min())
        hg.fit()
    elif(metric_name=='complete'):
        hg = HierarchicalGrouping(data_random, Max())
        hg.fit()
      
    elif(metric_name=='average'):
        hg = HierarchicalGrouping(data_random, AVG())
        hg.fit_with_avg()
    else:
        hg = HierarchicalGrouping(data_random, Min())
        

    print("------------------------------------------------- DATOS PARA AGRUPAR ------------------------------------------------------")
    for i,e in enumerate(data_random):
        print(" VALUE ", e, " INDEX ", i)
    print("------------------------------------------------- FIN DATOS PARA AGRUPAR ------------------------------------------------------")
   
    plot_dendogram(data_random,metric_name)
    


hg(data_set, 'average', 30)


""" def distance(element, element_to_compare):
    distances = [[abs(np.sqrt(sum([pow(x-vector_to_compare[i], 2) for i, x in enumerate(vector)])))
                  for vector in element] for vector_to_compare in element_to_compare]
    return distances 
        
data_set = [[[1,2,4,4]], [[5,6,7,8]], [[9,10,11,12]], [[5,2,4,5]]]


def calculate_distances(data_set):
    distances = [[0 for j in data_set] for i in data_set]
    for i in range(0,len(data_set)):
        for j in range(0,len(data_set)):
            distances[i][j]=distance(data_set[i], data_set[j])
    return distances
          
n = len(data_set)

def compare_min(distances):
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

def compare_max(distances):
    distances_max = [[ [max(dist) for dist in group] for group in groups]for groups in distances]

    dist_max = None
    
    for i in range(0,len(distances_max)):
        for j in range(0, i):
          if(dist_max == None):
            dist_max = max(distances_max[i][j])

            col = i
            row = j
          elif(distances_max[i][j] < dist_max):
            dist_max=max(distances_max[i][j])
            col = i
            row = j
    return col, row
#def compare_prom(distances):
    
print("data", data_set)
for _ in range(0,n-1):
   
    distances = calculate_distances(data_set)
    i, j, dist =  compare_min(distances)
    print("Fila: ",data_set[i],"Merge: ",data_set[j],"Dist: ", dist )
    data_set[i] = data_set[i] + data_set[j]
    
    data_set.pop(j)
    n-=1 """
""" HierarchicalGrouping(data_set, Min()).fit()

def plot_dendogram(data_set):        
    dendrogram = sch.dendrogram(sch.linkage(data_set, method='single'))
    
    plt.title('Dendograma')
    plt.xlabel('Indice - Fila')
    plt.ylabel('Distancia')
    plt.show()

plot_dendogram([[1,2,4,4], [5,6,7,8], [9,10,11,12], [5,2,4,5]])

kn = Kohonen(data_set,4)
kn.fit() """