
import networkx as nx
from heapq import heappop, heappush
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import unittest

# Función para crear un modelo de grafo que represente el sistema de transporte público.
def crear_grafo():
    """
    Esta función crea un objeto de tipo `nx.DiGraph` de la biblioteca `networkx` para representar la red de transporte público.

    Retorna:
        Un objeto `nx.DiGraph` que representa la red de transporte público.
    """
    grafo = nx.DiGraph()
    grafo.add_edge("Estación A", "Estación B", tiempo=10, costo=20, linea="Línea 1")
    grafo.add_edge("Estación B", "Estación C", tiempo=15, costo=30, linea="Línea 1")
    grafo.add_edge("Estación C", "Estación D", tiempo=20, costo=40, linea="Línea 2")
    # Agregar más conexiones según sea necesario
    return grafo

# Función para implementar el algoritmo de Dijkstra mejorado.
def dijkstra_mejorado(grafo, origen, destino, criterio):
    """
    Esta función implementa el algoritmo de Dijkstra mejorado para encontrar la ruta más corta entre dos estaciones en un grafo dirigido, considerando un criterio específico (tiempo o costo).

    Parámetros:
        grafo: Un objeto `nx.DiGraph` que representa la red de transporte público.
        origen: La estación de origen del viaje.
        destino: La estación de destino del viaje.
        criterio: El criterio a considerar para la búsqueda de la ruta más corta ("tiempo" o "costo").

    Retorna:
        La distancia mínima entre la estación de origen y la estación de destino, considerando el criterio especificado.
    """
    distancia = {estacion: float('inf') for estacion in grafo.nodes()}
    distancia[origen] = 0
    pq = [(0, origen)]

    while pq:
        _, estacion_actual = heappop(pq)

        if estacion_actual == destino:
            return distancia[destino]

        for vecino, datos in grafo[estacion_actual].items():
            nueva_distancia = distancia[estacion_actual] + datos[criterio]

            if nueva_distancia < distancia[vecino]:
                distancia[vecino] = nueva_distancia
                heappush(pq, (nueva_distancia, vecino))

# Función para cargar y preprocesar los datos de demanda de pasajeros.
def cargar_y_preprocesar_datos_demanda(ruta_archivo):
    """
    Esta función carga los datos de demanda de pasajeros desde un archivo CSV, los preprocesa y los devuelve como DataFrames de Pandas.

    Parámetros:
        ruta_archivo: La ruta del archivo CSV que contiene los datos de demanda de pasajeros.

    Retorna:
        Un DataFrame de Pandas con los datos de demanda de pasajeros preprocesados.
    """
    datos_demanda = pd.read_csv(ruta_archivo)

    # Preprocesamiento de datos
    # ... (Aplicar las técnicas de preprocesamiento necesarias, como limpieza, transformación y selección de características)

    return datos_demanda

# Función para entrenar un modelo de regresión lineal para predecir la demanda de pasajeros.
def entrenar_modelo_demanda(datos_demanda):
    """
    Esta función entrena un modelo de regresión lineal para predecir la demanda de pasajeros en una ruta específica.

    Parámetros:
        datos_demanda: Un DataFrame de Pandas con los datos de demanda de pasajeros preprocesados.

    Retorna:
        Un objeto `LinearRegression` entrenado para predecir la demanda de pasajeros.
    """
    X = datos_demanda[["hora_viaje", "costo_viaje", "linea_viaje", "dia_semana", "evento_especial", "condiciones_climaticas"]]
    y = datos_demanda["demanda"]

    # Normalizar X
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_demanda = LinearRegression()
    modelo_demanda.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = modelo_demanda.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio: {mse}")

    return modelo_demanda

# Función para predecir la demanda de pasajeros en una ruta específica utilizando el modelo entrenado.
def predecir_demanda(modelo_demanda, datos_ruta):
    """
    Esta función predice la demanda de pasajeros en una ruta específica utilizando el modelo entrenado.

    Parámetros:
        modelo_demanda: Un objeto `LinearRegression` entrenado para predecir la demanda de pasajeros.
        datos_ruta: Un DataFrame de Pandas con los datos de la ruta para la cual se desea predecir la demanda.

    Retorna:
        La demanda de pasajeros predicha para la ruta especificada.
    """
    X = datos_ruta[["hora_viaje", "costo_viaje", "linea_viaje", "dia_semana", "evento_especial", "condiciones_climaticas"]]
    demanda_predicha = modelo_demanda.predict(X)

    return demanda_predicha

