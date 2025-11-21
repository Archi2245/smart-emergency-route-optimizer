# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, json

from src.synthetic_graph import generate_graph, simulate_historical, save_graph_csv
from src.ml_model import load_model_and_scaler, predict_congestion_keras, train_model
from src.route_algorithms import dijkstra_time, a_star_time
from src.ga_optimizer import ga_optimize_route
from src.utils import edges_to_json

app = Flask(__name__)
CORS(app)

# In-memory state
GRAPH_STATE = {
    'coords': None,
    'edges': None,
    'graph': None
}
MODEL = None
SCALER = None

@app.route('/api/graph/generate', methods=['POST'])
def api_generate_graph():
    params = request.json or {}
    n_nodes = int(params.get('n_nodes', 20))
    n_edges = int(params.get('n_edges', 60))
    coords, edges, graph = generate_graph(n_nodes=n_nodes, n_edges=n_edges)
    GRAPH_STATE['coords'] = coords
    GRAPH_STATE['edges'] = edges
    GRAPH_STATE['graph'] = graph
    save_graph_csv(edges, path='data/synthetic_graph.csv')
    return jsonify({'status':'ok', 'n_nodes': len(coords), 'n_edges': len(edges), 'coords_sample': list(coords.items())[:5]})

@app.route('/api/graph/coords', methods=['GET'])
def api_get_coords():
    if GRAPH_STATE['coords'] is None:
        return jsonify({'status':'error','message':'No graph generated'}), 400
    # return coords as mapping node->(x,y)
    return jsonify({'status':'ok','coords': GRAPH_STATE['coords']})

@app.route('/api/model/train', methods=['POST'])
def api_train_model():
    if GRAPH_STATE['edges'] is None:
        return jsonify({'status':'error','message':'Generate graph first'}), 400
    params = request.json or {}
    epochs = int(params.get('epochs', 25))
    df_hist = simulate_historical(GRAPH_STATE['edges'], days=30)
    model, scaler = train_model(df_hist, epochs=epochs)
    global MODEL, SCALER
    MODEL = model; SCALER = scaler
    return jsonify({'status':'ok','message':'model trained', 'samples': len(df_hist)})

@app.route('/api/model/load', methods=['POST'])
def api_model_load():
    global MODEL, SCALER
    try:
        MODEL, SCALER = load_model_and_scaler()
        return jsonify({'status':'ok','message':'model loaded'})
    except Exception as e:
        return jsonify({'status':'error','message': str(e)}), 500

@app.route('/api/route/basic', methods=['POST'])
def api_route_basic():
    data = request.json
    source = int(data['source']); target = int(data['target'])
    alg = data.get('algorithm', 'dijkstra')
    hour = int(data.get('hour', 9)); day = int(data.get('day', 2)); is_holiday = int(data.get('is_holiday', 0))
    if GRAPH_STATE['graph'] is None:
        return jsonify({'status':'error','message':'Graph not generated'}), 400
    edges = GRAPH_STATE['edges']
    graph = GRAPH_STATE['graph']
    coords = GRAPH_STATE['coords']
    # set up predict_fn using loaded model
    if MODEL is None or SCALER is None:
        return jsonify({'status':'error','message':'Model not loaded or trained'}), 400
    def predict_fn(u,v):
        key = (u,v) if (u,v) in edges else (v,u)
        meta = edges[key]
        lvl, probs = predict_congestion_keras(MODEL, SCALER, meta, hour, day, is_holiday)
        return int(lvl)
    if alg == 'dijkstra':
        path, time_min = dijkstra_time(graph, predict_fn, source, target)
    else:
        path, time_min = a_star_time(coords, graph, predict_fn, source, target)
    return jsonify({'status':'ok','algorithm':alg,'path': path, 'estimated_time_min': time_min})

@app.route('/api/route/ga', methods=['POST'])
def api_route_ga():
    data = request.json
    source = int(data['source']); target = int(data['target'])
    hour = int(data.get('hour', 9)); day = int(data.get('day', 2)); is_holiday = int(data.get('is_holiday', 0))
    if GRAPH_STATE['edges'] is None:
        return jsonify({'status':'error','message':'Graph not generated'}), 400
    if MODEL is None or SCALER is None:
        return jsonify({'status':'error','message':'Model not loaded/trained'}), 400
    coords = GRAPH_STATE['coords']
    edges_map = GRAPH_STATE['edges']
    # We'll create a snapshot dict (simple copy of edges) to evaluate GA on latest base congestion
    snapshot_edges = {k:v.copy() for k,v in edges_map.items()}
    # define predict_fn_for_snapshot using MODEL+SCALER and snapshot meta
    def predict_fn_for_snapshot(u,v):
        key = (u,v) if (u,v) in snapshot_edges else (v,u)
        meta = snapshot_edges[key]
        lvl, probs = predict_congestion_keras(MODEL, SCALER, meta, hour, day, is_holiday)
        return int(lvl)
    mask, best_score, best_path = ga_optimize_route(coords, edges_map, source, target, predict_fn_for_snapshot,
                                                   pop_size=20, gens=25, snapshot_edges=snapshot_edges)
    return jsonify({'status':'ok','path': best_path, 'estimated_time_min': best_score})

@app.route('/api/snapshots', methods=['GET'])
def api_get_snapshots():
    if GRAPH_STATE['edges'] is None:
        return jsonify({'status':'error','message':'Graph not generated'}), 400
    # For frontend we return a single snapshot (list of edges)
    return jsonify({'status':'ok','snapshot': edges_to_json(GRAPH_STATE['edges'])})

@app.route('/api/graph/csv', methods=['GET'])
def api_graph_csv():
    path = 'data/synthetic_graph.csv'
    if os.path.exists(path):
        return send_file(path, mimetype='text/csv', as_attachment=True)
    return jsonify({'status':'error','message':'file not found'}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
