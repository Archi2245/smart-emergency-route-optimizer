# src/synthetic_graph.py
import numpy as np
import pandas as pd
import math

def generate_graph(n_nodes=20, n_edges=60, seed=42):
    """
    Generate a synthetic traffic network graph.
    
    Returns:
        coords: dict {node_id: (x, y)}
        edges: dict {(u,v): {'distance_km', 'base_speed_kmh', 'base_congestion'}}
        graph: adjacency list {node: [(neighbor, edge_key), ...]}
    """
    np.random.seed(seed)
    
    # Generate node coordinates in a 100x100 grid
    coords = {i: (np.random.uniform(0, 100), np.random.uniform(0, 100)) 
              for i in range(n_nodes)}
    
    edges = {}
    graph = {i: [] for i in range(n_nodes)}
    
    # Create edges - ensure connectivity first with MST-like approach
    connected = {0}
    unconnected = set(range(1, n_nodes))
    
    # Connect all nodes first
    while unconnected:
        u = np.random.choice(list(connected))
        v = np.random.choice(list(unconnected))
        
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        edge_key = (min(u,v), max(u,v))
        edges[edge_key] = {
            'distance_km': round(dist / 10, 2),  # Scale to reasonable km
            'base_speed_kmh': np.random.choice([30, 40, 50, 60]),
            'base_congestion': np.random.choice([0, 1, 2])  # 0=low, 1=med, 2=high
        }
        
        graph[u].append((v, edge_key))
        graph[v].append((u, edge_key))
        
        connected.add(v)
        unconnected.remove(v)
    
    # Add additional random edges
    remaining_edges = n_edges - (n_nodes - 1)
    attempts = 0
    max_attempts = remaining_edges * 10
    
    while len(edges) < n_edges and attempts < max_attempts:
        u, v = np.random.choice(n_nodes, 2, replace=False)
        edge_key = (min(u,v), max(u,v))
        
        if edge_key not in edges:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            edges[edge_key] = {
                'distance_km': round(dist / 10, 2),
                'base_speed_kmh': np.random.choice([30, 40, 50, 60]),
                'base_congestion': np.random.choice([0, 1, 2])
            }
            
            graph[u].append((v, edge_key))
            graph[v].append((u, edge_key))
        
        attempts += 1
    
    print(f"Generated graph: {n_nodes} nodes, {len(edges)} edges")
    return coords, edges, graph


def simulate_historical(edges, days=30, samples_per_edge=100):
    """
    Simulate historical traffic data for ML training.
    
    Returns:
        DataFrame with columns: hour, day_of_week, is_holiday, distance_km, 
                                base_speed_kmh, base_congestion, congestion_level
    """
    data = []
    
    for edge_key, meta in edges.items():
        for _ in range(samples_per_edge):
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            is_holiday = np.random.choice([0, 1], p=[0.9, 0.1])
            
            # Simulate congestion based on time patterns
            base_cong = meta['base_congestion']
            
            # Peak hours (7-9 AM, 5-7 PM) increase congestion
            if hour in [7, 8, 17, 18]:
                congestion_prob = [0.2, 0.3, 0.5]
            elif hour in [9, 10, 11, 12, 13, 14, 15, 16]:
                congestion_prob = [0.4, 0.4, 0.2]
            else:
                congestion_prob = [0.7, 0.2, 0.1]
            
            # Weekends less congestion
            if day_of_week >= 5:
                congestion_prob = [0.6, 0.3, 0.1]
            
            # Holidays even less
            if is_holiday:
                congestion_prob = [0.8, 0.15, 0.05]
            
            congestion_level = np.random.choice([0, 1, 2], p=congestion_prob)
            
            data.append({
                'hour': hour,
                'day_of_week': day_of_week,
                'is_holiday': is_holiday,
                'distance_km': meta['distance_km'],
                'base_speed_kmh': meta['base_speed_kmh'],
                'base_congestion': base_cong,
                'congestion_level': congestion_level
            })
    
    df = pd.DataFrame(data)
    print(f"Simulated {len(df)} historical traffic samples")
    return df


def save_graph_csv(edges, path='data/synthetic_graph.csv'):
    """Save graph edges to CSV file."""
    import os
    os.makedirs('data', exist_ok=True)
    
    rows = []
    for (u, v), meta in edges.items():
        rows.append({
            'node_u': u,
            'node_v': v,
            'distance_km': meta['distance_km'],
            'base_speed_kmh': meta['base_speed_kmh'],
            'base_congestion': meta['base_congestion']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved graph to {path}")