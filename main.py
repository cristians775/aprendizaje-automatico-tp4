import collections
import copy
from math import nan
import math
from random import randint
import pandas as pd
import numpy as np

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



                     
    

standardize_attributes(data_set,[ 'age', 'cad.dur', 'choleste' ])


#Definimos un numero para k
k = 3

class KMeans():
    def __init__(self,k, data_set) -> None:
        super().__init__()
        self.data_set = data_set
        self.k = k

    def fit(self):
        # Clonamos el data_set para poder modificar data_set_copy y no tener problemas de referencias
        
        #Asignamos aleatoriamente un numero de clase de 1 a k a cada una de las observaciones  -> [1,2,3,k] 
        for element in self.data_set:
            class_random = randint(0,k)
            element.append(class_random)
        data_set_update = copy.deepcopy(self.data_set)
        previous_data_set_update = []
        
        # Comparamos con el anterior antes de updatear asi podemos comprobar si todos tienen la misma clase
        while(not all(self.compare_all(data_set_update, previous_data_set_update))):
            # Clonamos el data_set_anterior
            previous_data_set_update = copy.deepcopy(data_set_update) 
            
            # Calculamos los centroides
            centroids = self.centroids(data_set_update)
            
            # Calculamos la distancia de los elemenos con los centroides mas cercanos
            for i, element in enumerate(self.data_set):
                # Calculamos la distancia entre el punto y todos los centroides y nos quedamos con la minima
                cluster = self.get_nearest_cluster(element, centroids)
                # Reemplazamos el elemento por el centroide que tiene mas cerca
                data_set_update[i][len(element)-1]=cluster
                

        
        return data_set_update
        
            
       
        
    def centroids(self, data_set):
        # Calculamos los centroides de cada clase
        centroids = []
        for cluster in range(self.k):
        #Obtenemos los elementos de la clase k
            # Filtramos por la clase k
            data_cluster = list(filter(lambda element: element[len(element)-1] == cluster, data_set))
            length_cluster = len(data_cluster[0])
            # Creamos un vector de ceros
            centroid = [0 for i in range(length_cluster)]
            # Le asignamos la clase al centroide -> [ 1, 2, 3, k ]
            centroid[length_cluster-1] = cluster
            #  Calculamos el centroide para la respectiva clase
            #  Ignoramos el ultimo elemento( valor de la clase ) de cada vector
            for cluster_element in data_cluster:
                # Sumamos todos los elementos de la columna i Ej: result = 4 en i= 1 -> [ [ 1, 2, 3 ], [1, 2, 3] ]
                for i in range(len(cluster_element)-2):
                    centroid[i] = sum(map(lambda element: element[i],data_cluster)) / len(data_cluster)
            centroids.append(centroid)
        return centroids
                
    def get_nearest_cluster(self, _element, centroids):
        # No tomamos en cuenta la clase para calcular la distancia asi que le quitamos el ultimo elemento
        element = _element[:len(_element)-1]
        # Calculamos las distancias -> elemento del vector -> abs(raiz_cuadrada(suma(potencia_cuadrada(xi-cluster[i]))))
        distances = [abs(np.sqrt(sum([ pow(x-cluster[i],2) for i,x in enumerate(element)]))) for cluster in centroids]
        return distances.index(min(distances))
            
        
    def compare_all(self, data_set, data_to_compare):
        # Si el len es diferente devolvemos un array con un False para que no se rompa el metodo all
        if len(data_to_compare) != len(data_set):
            return [ False ]
        else:
            # Comparamos todos los puntos del data_set con el anterior antes de updatear
            return [ all( [ ele == data_to_compare[i][j]for j, ele in enumerate(element)])for i, element in enumerate(data_set)]
        
kmeans = KMeans(k, data_set)
print(kmeans.fit())




        
