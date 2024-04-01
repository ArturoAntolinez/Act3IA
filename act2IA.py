import networkx as nx
from heapq import heappop, heappush

# Definir la base de conocimiento del sistema de transporte masivo
def crear_grafo():
    grafo = nx.DiGraph()
    grafo.add_edge("Estación A", "Estación B", tiempo=10, costo=20, linea="Línea 1")
    grafo.add_edge("Estación B", "Estación C", tiempo=15, costo=30, linea="Línea 1")
    grafo.add_edge("Estación C", "Estación D", tiempo=20, costo=40, linea="Línea 2")
    # Agregar más conexiones según sea necesario
    return grafo

# Implementar el algoritmo de Dijkstra mejorado
def dijkstra_mejorado(grafo, origen, destino, criterio):
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

# Crear el grafo del sistema de transporte masivo
grafo = crear_grafo()

# Calcular la ruta de menor costo basada en el tiempo de viaje
ruta_tiempo = dijkstra_mejorado(grafo, "Estación A", "Estación D", criterio="tiempo")
print(f"Ruta de menor tiempo de viaje: {ruta_tiempo} minutos")

# Calcular la ruta de menor costo basada en el costo monetario
ruta_costo = dijkstra_mejorado(grafo, "Estación A", "Estación D", criterio="costo")
print(f"Ruta de menor costo monetario: {ruta_costo} pesos")
