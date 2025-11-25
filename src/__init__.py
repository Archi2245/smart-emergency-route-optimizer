# src/__init__.py
"""
Smart Emergency Route Optimizer - Source Package
"""
__version__ = "1.0.0"

from .synthetic_graph import generate_graph, simulate_historical, save_graph_csv
from .ml_model import (
    train_model, 
    load_model_and_scaler, 
    predict_congestion_keras, 
    estimate_travel_time
)
from .route_algorithms import dijkstra_time, a_star_time, calculate_path_time
from .ga_optimizer import ga_optimize_route
from .utils import edges_to_json

__all__ = [
    'generate_graph',
    'simulate_historical',
    'save_graph_csv',
    'train_model',
    'load_model_and_scaler',
    'predict_congestion_keras',
    'estimate_travel_time',
    'dijkstra_time',
    'a_star_time',
    'calculate_path_time',
    'ga_optimize_route',
    'edges_to_json'
]