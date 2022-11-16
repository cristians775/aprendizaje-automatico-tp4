
import copy
from json.encoder import INFINITY
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from HierarchicalGrouping_2 import AVG, HierarchicalGrouping, Max, Min
from sklearn.cluster import KMeans as kms
import matplotlib.pyplot as plt
from KMeans import KMeans
from Kohonen import Kohonen
import scipy.cluster.hierarchy as sch

df = pd.read_excel("./data.xlsx")
df.fillna(0, inplace=True)
# Conjunto de datos
data_set = df.values.tolist()

# Filtramos los datos y quitamos las filas que tienen valores de los atributos en blanco
data_set = list(filter(lambda element: element[3]!=0, data_set))

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
    sgdiz_result = []
    sgdiz_column = list(map(lambda x: x[attibutes_dic['sigdz']],data_set))
    for i,result in enumerate(kmeans_result):
        sgdiz_result.append([sgdiz_column[i],result[len(result)-1]])
   #sgdiz_result-> [[sgdiz_value, cluster_value]]
    clusters = []
    #accuracy = calculate_accuracy(list(map(map(lambda x: x[k_attribute_position],data)), list(map(lambda x: x[k_attribute_position],kmeans_result))))
    for i in range(0,k):
        list_k = list(filter(lambda x: x[1] == i,sgdiz_result))
        len_sgdiz_0 = len(list(filter(lambda x: x[0]==0,list_k)))
        len_sgdiz_1 = len(list(filter(lambda x: x[0]==1,list_k)))
        
        if(len_sgdiz_0 > len_sgdiz_1):
            clusters.append({'cluster': i,'grupo':'sanos', 'cantidad de sanos': len_sgdiz_0, 'cantidad de enfermos':len_sgdiz_1})
        else:
            clusters.append({'cluster': i,'grupo':'enfermos', 'cantidad de sanos': len_sgdiz_0, 'cantidad de enfermos':len_sgdiz_1})

    #probamos 10 con datos random del data_set
    #mostramos los datos random
    random_data = get_data_random(new_data, 10)
    
    for result in kmeans.predict(random_data):

        print('Individuo: ', result,  ' pertenece al grupo de ', clusters[result[len(result)-1]]['grupo'])
    
    

# kmeans(8, data_set)

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

   
    plot_dendogram(data_random,metric_name)
    


#hg(data_set, 'complete', 30)

new_data = [[] for i in data_set]
for i, vector in enumerate(data_set):
# No tomamos en cuenta el atributo tvdlm
    new_data[i] = vector[:len(vector)-2]

kn = Kohonen(new_data,list(map(lambda x: x[attibutes_dic['sigdz']],data_set)), 10)
labels, w = kn.fit()


print("------ ETIQUETAS --------")
for i in range(0,len(labels)):
    for j in range(0, len(labels)):
        if(labels[i][j]['enfermo']>labels[i][j]['sano']):
            print('E', end=" , ")
        elif(labels[i][j]['enfermo']<labels[i][j]['sano']):
            print('S', end=" , ")
        
    print("")