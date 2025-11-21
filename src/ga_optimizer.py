# backend/src/ga_optimizer.py
import random
import numpy as np
from copy import deepcopy
import math
from heapq import heappush, heappop

# GA implementation that uses the graph (edges) provided by caller

def run_astar_on_filtered_graph(mask, edge_keys, snapshots_edges, coords, predict_fn_for_snapshot, source, target):
    # mask: binary list for edge_keys
    # snapshots_edges: a dict of canonical edges (we use the one snapshot passed in via predict_fn)
    # Build filtered graph
    fgraph = {i: [] for i in coords.keys()}
    for i,keep in enumerate(mask):
        if keep:
            a,b = edge_keys[i]
            meta = snapshots_edges[(a,b)]
            fgraph[a].append((b, meta))
            fgraph[b].append((a, meta))
    # A* local
    def heuristic_local(u,v):
        ux,uy = coords[u]; vx,vy = coords[v]
        straight_km = math.hypot(ux-vx, uy-vy)/10.0
        avg_speed = 40.0
        return straight_km / avg_speed * 60.0
    g_score = {i: float('inf') for i in fgraph.keys()}
    g_score[source] = 0.0
    open_set = []
    heappush(open_set, (heuristic_local(source,target), 0, source))
    came_from = {}
    visited = set()
    while open_set:
        f,g,u = heappop(open_set)
        if u == target:
            break
        if u in visited: continue
        visited.add(u)
        for (v, meta) in fgraph[u]:
            lvl = predict_fn_for_snapshot(u,v)  # uses snapshot features or model
            # tmin from meta
            if meta is None:
                tmin = 1e9
            else:
                if lvl == 0:
                    mult = 1.0
                elif lvl == 1:
                    mult = 1.4
                else:
                    mult = 1.9
                speed = meta['base_speed_kmph']/mult
                if speed <= 0.01: speed = 0.01
                tmin = meta['distance_km']/speed*60.0
            tentative = g_score[u] + tmin
            if tentative < g_score[v]:
                g_score[v] = tentative
                came_from[v] = u
                heappush(open_set, (tentative + heuristic_local(v,target), tentative, v))
    if g_score[target] == float('inf'):
        return None, float('inf')
    # reconstruct
    path = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = came_from.get(cur, None)
        if cur is None:
            return None, float('inf')
    path.append(source)
    path.reverse()
    return path, g_score[target]


def ga_optimize_route(coords, edges_map, source, target, predict_fn_for_snapshot,
                      pop_size=20, gens=25, tournament_k=3,
                      mutation_prob=0.08, crossover_prob=0.9, snapshot_edges=None):
    # edges_map: dictionary of canonical (a,b)->meta for the base graph
    edge_keys = list(edges_map.keys())
    E = len(edge_keys)

    def random_mask(p=0.85):
        return [1 if random.random() < p else 0 for _ in range(E)]

    def tournament_selection(pop, scores):
        ixs = random.sample(range(len(pop)), tournament_k)
        best = min(ixs, key=lambda i: scores[i])
        return deepcopy(pop[best])

    def crossover_mask(a, b):
        if random.random() > crossover_prob:
            return deepcopy(a)
        child = [a[i] if random.random() < 0.5 else b[i] for i in range(E)]
        return child

    def mutate_mask(mask):
        for i in range(E):
            if random.random() < mutation_prob:
                mask[i] = 1 - mask[i]
        return mask

    # initialize population
    pop = [random_mask() for _ in range(pop_size-1)]
    pop.append([1]*E)  # ensure all-ones exists
    best_mask = None; best_score = 1e12; best_path = None
    for g in range(gens):
        scores = []
        paths = []
        for ind in pop:
            score, path = run_astar_on_filtered_graph(ind, edge_keys, snapshot_edges, coords, predict_fn_for_snapshot, source, target)
            if score is None:
                score_val = 1e9
            else:
                score_val = score
            scores.append(score_val); paths.append(path)
        idx_best = int(np.argmin(scores))
        if scores[idx_best] < best_score:
            best_score = scores[idx_best]
            best_mask = deepcopy(pop[idx_best])
            best_path = paths[idx_best]
        # new pop (elitism of top 2)
        new_pop = []
        best_two_ix = np.argsort(scores)[:2]
        new_pop.append(deepcopy(pop[best_two_ix[0]]))
        new_pop.append(deepcopy(pop[best_two_ix[1]]))
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, scores)
            p2 = tournament_selection(pop, scores)
            child = crossover_mask(p1, p2)
            child = mutate_mask(child)
            new_pop.append(child)
        pop = new_pop
    return best_mask, best_score, best_path
