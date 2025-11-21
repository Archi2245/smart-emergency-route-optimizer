# backend/src/utils.py
import json
def edges_to_json(edges):
    out = []
    for (u,v), m in edges.items():
        out.append({'u':int(u), 'v':int(v), 'distance': m['distance_km'], 'base_speed': m['base_speed_kmph'], 'base_congestion': m['base_congestion']})
    return out
