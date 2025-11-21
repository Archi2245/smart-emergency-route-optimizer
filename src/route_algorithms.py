# backend/src/route_algorithms.py
import math
from heapq import heappush, heappop

def travel_time_minutes_from_meta(meta, congestion_level):
    if congestion_level == 0:
        mult = 1.0
    elif congestion_level == 1:
        mult = 1.4
    else:
        mult = 1.9
    speed = meta['base_speed_kmph'] / mult
    if speed <= 0.01: speed = 0.01
    return meta['distance_km'] / speed * 60.0

def dijkstra_time(graph, predict_fn, source, target):
    dist = {i: float('inf') for i in graph.keys()}
    prev = {i: None for i in graph.keys()}
    dist[source] = 0.0
    pq = [(0.0, source)]
    visited = set()
    while pq:
        d,u = heappop(pq)
        if u in visited: continue
        visited.add(u)
        if u == target: break
        for (v, meta) in graph[u]:
            lvl = predict_fn(u, v)
            tmin = travel_time_minutes_from_meta(meta, lvl)
            nd = d + tmin
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))
    if dist[target] == float('inf'):
        return None, float('inf')
    # reconstruct path
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[target]

def heuristic(coords, u, v):
    ux,uy = coords[u]; vx,vy = coords[v]
    straight_km = math.hypot(ux-vx, uy-vy)/10.0
    avg_speed = 40.0
    return straight_km / avg_speed * 60.0

def a_star_time(coords, graph, predict_fn, source, target):
    open_set = []
    heappush(open_set, (0 + heuristic(coords, source, target), 0, source))
    came_from = {}
    g_score = {i: float('inf') for i in graph.keys()}
    g_score[source] = 0.0
    visited = set()
    while open_set:
        f,g,u = heappop(open_set)
        if u == target:
            break
        if u in visited: continue
        visited.add(u)
        for (v, meta) in graph[u]:
            lvl = predict_fn(u, v)
            tmin = travel_time_minutes_from_meta(meta, lvl)
            tentative = g_score[u] + tmin
            if tentative < g_score[v]:
                g_score[v] = tentative
                came_from[v] = u
                fscore = tentative + heuristic(coords, v, target)
                heappush(open_set, (fscore, tentative, v))
    if g_score[target] == float('inf'):
        return None, float('inf')
    path = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = came_from.get(cur, None)
        if cur is None:
            break
    path.append(source)
    path.reverse()
    return path, g_score[target]
