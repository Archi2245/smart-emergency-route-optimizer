# backend/src/synthetic_graph.py
import math, random
import pandas as pd

def generate_graph(n_nodes=20, n_edges=60, seed=42):
    random.seed(seed)
    coords = {i: (random.uniform(0,100), random.uniform(0,100)) for i in range(n_nodes)}
    edges = {}
    edge_list = []
    while len(edge_list) < n_edges:
        a = random.randrange(n_nodes); b = random.randrange(n_nodes)
        if a==b: continue
        if (a,b) in edges or (b,a) in edges: continue
        dist = math.hypot(coords[a][0]-coords[b][0], coords[a][1]-coords[b][1]) / 10.0
        base_speed = random.uniform(30,60)
        base_congestion = random.choice([0.1,0.2,0.3,0.5])
        edges[(a,b)] = {
            'distance_km': round(dist,3),
            'base_speed_kmph': round(base_speed,2),
            'base_congestion': base_congestion
        }
        edge_list.append((a,b))
    # build adjacency (undirected)
    graph = {i: [] for i in range(n_nodes)}
    for (a,b),meta in edges.items():
        graph[a].append((b, meta))
        graph[b].append((a, meta))
    return coords, edges, graph

def simulate_historical(edges, days=30):
    import random
    hist_rows = []
    for (a,b), m in edges.items():
        for day in range(days):
            for hour in [6,9,12,15,18,21]:
                hour_factor = 0.8 if hour in [9,18] else 0.5 if hour in [6,21] else 0.3
                noise = random.uniform(-0.2,0.2)
                cong_score = m['base_congestion'] + hour_factor + noise
                if cong_score < 0.5:
                    level = 0
                elif cong_score < 0.9:
                    level = 1
                else:
                    level = 2
                hist_rows.append({
                    'u': a, 'v': b,
                    'hour': hour, 'day': day % 7,
                    'is_holiday': int(day%7 in (5,6) and random.random()<0.3),
                    'base_speed': m['base_speed_kmph'],
                    'distance': m['distance_km'],
                    'base_congestion': m['base_congestion'],
                    'congestion_level': level
                })
    return pd.DataFrame(hist_rows)

def save_graph_csv(edges, path='data/synthetic_graph.csv'):
    import os
    os.makedirs('data', exist_ok=True)
    rows = []
    for (a,b), m in edges.items():
        rows.append({'u': a, 'v': b, 'distance_km': m['distance_km'], 'base_speed_kmph': m['base_speed_kmph'], 'base_congestion': m['base_congestion']})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
