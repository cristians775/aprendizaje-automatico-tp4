import copy
from random import randint
import numpy as np

class KMeans():
    def __init__(self,k, data_set) -> None:
        super().__init__()
        self.data_set = copy.deepcopy(data_set)
        self.k = k

    def fit(self):
        #Asignamos aleatoriamente un numero de clase de 1 a k a cada una de las observaciones  -> [1,2,3,k] 

           
        for i in range(0,len(self.data_set)):
            #print('i', len(i))
            self.data_set[i]= self.data_set[i] + [randint(0,self.k)]
            

        
        # Clonamos el data_set para poder modificar data_set_copy y no tener problemas de referencias
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
        length_cluster = len(self.data_set[0])
        for cluster in range(0,self.k):
            #Obtenemos los elementos de la clase k
            # Filtramos por la clase k
            data_cluster = list(filter(lambda element: element[len(element)-1] == cluster, data_set))
           
            # Creamos un vector de ceros
            centroid = [0 for i in range(0,length_cluster-1)]
   
            for cluster_element in data_cluster:
                # Sumamos todos los elementos de la columna i Ej: result = 4 en i= 1 -> [ [ 1, 2,3 ], [1, 2, 3] ]
                for i in range(0,len(cluster_element)-1):
                    
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