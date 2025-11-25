# src/route_algorithms.py
import heapq
import math
from src.ml_model import estimate_travel_time

def dijkstra_time(graph, predict_fn, source, target):
    """
    Dijkstra's algorithm for shortest path based on predicted travel time.
    
    Args:
        graph: adjacency list {node: [(neighbor, edge_key), ...]}
        predict_fn: function(u, v) -> congestion_level
        source: start node
        target: end node
    
    Returns:
        path: list of nodes
        total_time: total estimated time in minutes
    """
    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    previous = {node: None for node in graph}
    pq = [(0, source)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == target:
            break
        
        for neighbor, edge_key in graph[current]:
            if neighbor in visited:
                continue
            
            # Get predicted congestion
            u, v = edge_key
            congestion_level = predict_fn(u, v)
            
            # Calculate edge weight (travel time)
            # We need edge metadata - extract from edge_key
            # For now, use a simple heuristic
            time = 1 + congestion_level * 2  # Simple time estimation
            
            new_dist = current_dist + time
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct path
    if distances[target] == float('inf'):
        return None, None
    
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return path, distances[target]


def a_star_time(coords, graph, predict_fn, source, target):
    """
    A* algorithm with Euclidean distance heuristic.
    
    Heuristic: straight-line distance / average speed
    """
    def heuristic(node):
        x1, y1 = coords[node]
        x2, y2 = coords[target]
        euclidean_dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return euclidean_dist / 50  # Assume avg 50 km/h for heuristic
    
    g_score = {node: float('inf') for node in graph}
    g_score[source] = 0
    
    f_score = {node: float('inf') for node in graph}
    f_score[source] = heuristic(source)
    
    previous = {node: None for node in graph}
    open_set = [(f_score[source], source)]
    closed_set = set()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        if current == target:
            break
        
        for neighbor, edge_key in graph[current]:
            if neighbor in closed_set:
                continue
            
            # Get predicted congestion and calculate time
            u, v = edge_key
            congestion_level = predict_fn(u, v)
            time = 1 + congestion_level * 2
            
            tentative_g = g_score[current] + time
            
            if tentative_g < g_score[neighbor]:
                previous[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    if g_score[target] == float('inf'):
        return None, None
    
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return path, g_score[target]


def calculate_path_time(path, edges, predict_fn):
    """
    Calculate total travel time for a given path.
    
    Args:
        path: list of nodes
        edges: dict of edge metadata
        predict_fn: function to predict congestion
    
    Returns:
        total_time: total estimated time in minutes
    """
    from .ml_model import estimate_travel_time
    
    total_time = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_key = (min(u,v), max(u,v))
        
        if edge_key not in edges:
            continue
        
        meta = edges[edge_key]
        congestion_level = predict_fn(u, v)
        time = estimate_travel_time(
            meta['distance_km'],
            meta['base_speed_kmh'],
            congestion_level
        )
        total_time += time
    
    return total_time