# app.py - Flask Backend (Corrected Version)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import traceback

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
    """Generate synthetic traffic network and auto-train model"""
    global MODEL, SCALER
    try:
        params = request.json or {}
        n_nodes = int(params.get('n_nodes', 20))
        n_edges = int(params.get('n_edges', 60))
        
        print(f"[INFO] Generating graph with {n_nodes} nodes and {n_edges} edges...")
        coords, edges, graph = generate_graph(n_nodes=n_nodes, n_edges=n_edges)
        
        # Convert coords keys to strings for JSON serialization
        coords_str = {str(k): v for k, v in coords.items()}
        
        GRAPH_STATE['coords'] = coords  # Keep original for internal use
        GRAPH_STATE['edges'] = edges
        GRAPH_STATE['graph'] = graph
        
        # Save graph to CSV
        os.makedirs('data', exist_ok=True)
        save_graph_csv(edges, path='data/synthetic_graph.csv')
        
        # Auto-train model after generating graph
        print("[INFO] Training model automatically...")
        df_hist = simulate_historical(edges, days=30)
        model, scaler = train_model(df_hist, epochs=25)
        MODEL = model
        SCALER = scaler
        print("[SUCCESS] Model trained successfully!")
        
        coords_str = {str(k): v for k, v in GRAPH_STATE['coords'].items()}
        return jsonify({'status':'ok','coords': coords_str})
    except Exception as e:
        print(f"[ERROR] Error generating graph: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/graph/coords', methods=['GET'])
def api_get_coords():
    """Get node coordinates"""
    if GRAPH_STATE['coords'] is None:
        return jsonify({'status': 'error', 'message': 'No graph generated'}), 400
    
    # Convert to string keys for JSON
    coords_str = {str(k): v for k, v in GRAPH_STATE['coords'].items()}
    return jsonify({'status': 'ok', 'coords': coords_str})

@app.route('/api/model/train', methods=['POST'])
def api_train_model():
    """Train ML model on historical data"""
    global MODEL, SCALER
    if GRAPH_STATE['edges'] is None:
        return jsonify({'status': 'error', 'message': 'Generate graph first'}), 400
    try:
        params = request.json or {}
        epochs = int(params.get('epochs', 25))
        
        print(f"[INFO] Training model with {epochs} epochs...")
        df_hist = simulate_historical(GRAPH_STATE['edges'], days=30)
        model, scaler = train_model(df_hist, epochs=epochs)
        MODEL = model
        SCALER = scaler
        
        return jsonify({
            'status': 'ok',
            'message': 'Model trained successfully', 
            'samples': len(df_hist)
        })
    except Exception as e:
        print(f"[ERROR] Error training model: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/model/load', methods=['POST'])
def api_model_load():
    """Load pre-trained model from disk"""
    global MODEL, SCALER
    try:
        MODEL, SCALER = load_model_and_scaler()
        return jsonify({'status': 'ok', 'message': 'Model loaded from disk'})
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/route/basic', methods=['POST'])
def api_route_basic():
    """Compute route using Dijkstra or A* algorithm"""
    try:
        data = request.json
        source = int(data['source'])
        target = int(data['target'])
        alg = data.get('algorithm', 'dijkstra')
        hour = int(data.get('hour', 9))
        day = int(data.get('day', 2))
        is_holiday = int(data.get('is_holiday', 0))
        
        if GRAPH_STATE['graph'] is None:
            return jsonify({'status': 'error', 'message': 'Graph not generated'}), 400
        
        if MODEL is None or SCALER is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded or trained'}), 400
        
        edges = GRAPH_STATE['edges']
        graph = GRAPH_STATE['graph']
        coords = GRAPH_STATE['coords']
        
        # Validate source and target
        if source not in coords or target not in coords:
            return jsonify({
                'status': 'error',
                'message': f'Invalid nodes. Valid range: 0-{len(coords)-1}'
            }), 400
        
        # Prediction function for edge weights
        def predict_fn(u, v):
            key = (u, v) if (u, v) in edges else (v, u)
            if key not in edges:
                return 0
            meta = edges[key]
            lvl, probs = predict_congestion_keras(MODEL, SCALER, meta, hour, day, is_holiday)
            return int(lvl)
        
        print(f"[INFO] Computing route: {source} -> {target} using {alg}")
        
        # Compute route based on algorithm
        if alg == 'dijkstra':
            path, time_min = dijkstra_time(graph, predict_fn, source, target)
        elif alg == 'a_star' or alg == 'a*':
            path, time_min = a_star_time(coords, graph, predict_fn, source, target)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid algorithm'}), 400
        
        if path is None:
            return jsonify({'status': 'error', 'message': 'No path found'}), 400
        
        print(f"[SUCCESS] Route found: {path}, Time: {time_min:.2f} min")
        
        return jsonify({
            'status': 'ok',
            'algorithm': alg,
            'path': path, 
            'estimated_time_min': float(time_min)
        })
    except Exception as e:
        print(f"[ERROR] Error in route/basic: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/route/ga', methods=['POST'])
def api_route_ga():
    """Compute route using Genetic Algorithm"""
    try:
        data = request.json
        source = int(data['source'])
        target = int(data['target'])
        hour = int(data.get('hour', 9))
        day = int(data.get('day', 2))
        is_holiday = int(data.get('is_holiday', 0))
        
        if GRAPH_STATE['edges'] is None:
            return jsonify({'status': 'error', 'message': 'Graph not generated'}), 400
        
        if MODEL is None or SCALER is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded/trained'}), 400
        
        coords = GRAPH_STATE['coords']
        edges_map = GRAPH_STATE['edges']
        
        # Validate source and target
        if source not in coords or target not in coords:
            return jsonify({
                'status': 'error',
                'message': f'Invalid nodes. Valid range: 0-{len(coords)-1}'
            }), 400
        
        # Create snapshot of edges for GA
        snapshot_edges = {k: v.copy() for k, v in edges_map.items()}
        
        # Prediction function
        def predict_fn_for_snapshot(u, v):
            key = (u, v) if (u, v) in snapshot_edges else (v, u)
            if key not in snapshot_edges:
                return 0
            meta = snapshot_edges[key]
            lvl, probs = predict_congestion_keras(MODEL, SCALER, meta, hour, day, is_holiday)
            return int(lvl)
        
        print(f"[INFO] Computing route with GA: {source} -> {target}")
        
        # Run genetic algorithm
        mask, best_score, best_path = ga_optimize_route(
            coords, edges_map, source, target, predict_fn_for_snapshot,
            pop_size=20, gens=25, snapshot_edges=snapshot_edges
        )
        
        if best_path is None:
            return jsonify({'status': 'error', 'message': 'No path found by GA'}), 400
        
        print(f"[SUCCESS] GA route found: {best_path}, Time: {best_score:.2f} min")
        
        return jsonify({
            'status': 'ok',
            'algorithm': 'GA',
            'path': best_path, 
            'estimated_time_min': float(best_score)
        })
    except Exception as e:
        print(f"[ERROR] Error in route/ga: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/snapshots', methods=['GET'])
def api_get_snapshots():
    """Get current graph snapshot"""
    if GRAPH_STATE['edges'] is None:
        return jsonify({'status': 'error', 'message': 'Graph not generated'}), 400
    return jsonify({'status': 'ok', 'snapshot': edges_to_json(GRAPH_STATE['edges'])})

@app.route('/api/graph/csv', methods=['GET'])
def api_graph_csv():
    """Download graph as CSV file"""
    path = 'data/synthetic_graph.csv'
    if os.path.exists(path):
        return send_file(path, mimetype='text/csv', as_attachment=True, 
                        download_name='synthetic_graph.csv')
    return jsonify({'status': 'error', 'message': 'CSV file not found'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'graph_loaded': GRAPH_STATE['graph'] is not None,
        'model_loaded': MODEL is not None,
        'message': 'Backend is running'
    })

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'name': 'Smart Emergency Route Optimizer API',
        'version': '1.0.0',
        'endpoints': [
            'POST /api/graph/generate',
            'GET /api/graph/coords',
            'POST /api/model/train',
            'POST /api/route/basic',
            'POST /api/route/ga',
            'GET /api/health'
        ]
    })

if __name__ == "__main__":
    print("="*60)
    print("🚑 Smart Emergency Route Optimizer Backend")
    print("="*60)
    print("Starting Flask server...")
    print("Backend running on: http://localhost:5000")
    print("API health check: http://localhost:5000/api/health")
    print("="*60)
    app.run(debug=True, port=5000, host='0.0.0.0')