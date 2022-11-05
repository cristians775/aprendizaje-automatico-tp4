
import math

import pandas as pd
import numpy as np

from KMeans import KMeans

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


        
kmeans = KMeans(k, data_set)
print(kmeans.fit())




        
