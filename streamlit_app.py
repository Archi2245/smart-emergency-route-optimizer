import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Configuration
API_BASE = "http://localhost:5000/api"

# Page config
st.set_page_config(
    page_title="AI Traffic Routing System",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'coords' not in st.session_state:
    st.session_state.coords = None
if 'route_result' not in st.session_state:
    st.session_state.route_result = None

# Helper functions
def generate_graph(n_nodes, n_edges):
    try:
        response = requests.post(f"{API_BASE}/graph/generate", 
                                json={"n_nodes": n_nodes, "n_edges": n_edges})
        return response.json()
    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")
        return None

def get_coordinates():
    try:
        response = requests.get(f"{API_BASE}/graph/coords")
        if response.status_code == 200:
            return response.json()['coords']
    except Exception as e:
        st.error(f"Error fetching coordinates: {str(e)}")
    return None

def train_model(epochs):
    try:
        response = requests.post(f"{API_BASE}/model/train", 
                                json={"epochs": epochs})
        return response.json()
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def compute_route(source, target, algorithm, hour, day, is_holiday):
    try:
        endpoint = f"{API_BASE}/route/ga" if algorithm == "GA" else f"{API_BASE}/route/basic"
        payload = {
            "source": source,
            "target": target,
            "hour": hour,
            "day": day,
            "is_holiday": is_holiday
        }
        if algorithm != "GA":
            payload["algorithm"] = algorithm.lower()
        
        response = requests.post(endpoint, json=payload)
        return response.json()
    except Exception as e:
        st.error(f"Error computing route: {str(e)}")
        return None

def plot_graph_with_route(coords, path=None):
    """Create an interactive plotly visualization of the graph"""
    if not coords:
        return None
    
    # Convert coords to lists
    node_ids = list(coords.keys())
    node_x = [float(coords[str(n)][0]) for n in node_ids]
    node_y = [float(coords[str(n)][1]) for n in node_ids]
    
    fig = go.Figure()
    
    # Plot all nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=12, color='lightblue', line=dict(width=2, color='darkblue')),
        text=[str(n) for n in node_ids],
        textposition="top center",
        name='Nodes',
        hovertemplate='Node: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
    ))
    
    # Highlight route if provided
    if path and len(path) > 1:
        path_x = [float(coords[str(n)][0]) for n in path]
        path_y = [float(coords[str(n)][1]) for n in path]
        
        fig.add_trace(go.Scatter(
            x=path_x,
            y=path_y,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=15, color='red', symbol='circle'),
            name='Route',
            hovertemplate='Node: %{text}<extra></extra>',
            text=[str(n) for n in path]
        ))
    
    fig.update_layout(
        title="Traffic Network Graph",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='white'
    )
    
    return fig

# Title and description
st.title("🚗 AI-Powered Traffic Routing System")
st.markdown("""
**Machine Learning-based route optimization using Neural Networks and Genetic Algorithms**

This system demonstrates:
- **ANN Congestion Predictor**: Neural network trained on historical traffic patterns
- **Traditional Algorithms**: Dijkstra and A* pathfinding
- **Genetic Algorithm**: Evolutionary optimization for route selection
""")

# Sidebar for setup
st.sidebar.header("⚙️ System Configuration")

with st.sidebar:
    st.subheader("1. Generate Network")
    n_nodes = st.number_input("Number of Nodes", min_value=10, max_value=50, value=20)
    n_edges = st.number_input("Number of Edges", min_value=20, max_value=200, value=60)
    
    if st.button("🌐 Generate Graph", type="primary"):
        with st.spinner("Generating synthetic traffic network..."):
            result = generate_graph(n_nodes, n_edges)
            if result and result.get('status') == 'ok':
                st.session_state.graph_generated = True
                st.session_state.coords = get_coordinates()
                st.success(f"✅ Generated {result['n_nodes']} nodes and {result['n_edges']} edges")
    
    st.divider()
    
    st.subheader("2. Train ML Model")
    epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=25)
    
    if st.button("🧠 Train Neural Network", disabled=not st.session_state.graph_generated):
        with st.spinner("Training ANN on historical traffic data..."):
            result = train_model(epochs)
            if result and result.get('status') == 'ok':
                st.session_state.model_trained = True
                st.success(f"✅ Model trained on {result['samples']} samples")

# Main content area
if not st.session_state.graph_generated:
    st.info("👈 Start by generating a traffic network from the sidebar")
elif not st.session_state.model_trained:
    st.info("👈 Next, train the neural network model from the sidebar")
else:
    # Create two columns for route planning
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Route Planning")
        
        # Get node options
        node_options = sorted([int(k) for k in st.session_state.coords.keys()])
        
        source = st.selectbox("Source Node", node_options, index=0)
        target = st.selectbox("Target Node", node_options, index=min(len(node_options)-1, 5))
        
        st.subheader("⏰ Traffic Conditions")
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=9)
        day = st.selectbox("Day of Week", 
                          options=list(range(7)),
                          format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x],
                          index=2)
        is_holiday = st.checkbox("Holiday")
        
        st.subheader("🔍 Algorithm Selection")
        algorithm = st.radio(
            "Choose routing algorithm:",
            ["Dijkstra", "A*", "GA"],
            help="""
            - **Dijkstra**: Classic shortest path algorithm
            - **A***: Heuristic-based pathfinding (faster)
            - **GA**: Genetic Algorithm optimization
            """
        )
        
        if st.button("🚀 Compute Route", type="primary"):
            with st.spinner(f"Computing route using {algorithm}..."):
                result = compute_route(source, target, algorithm, hour, day, int(is_holiday))
                if result and result.get('status') == 'ok':
                    st.session_state.route_result = result
                    st.success("✅ Route computed successfully!")
    
    with col2:
        st.subheader("📊 Visualization")
        
        # Display graph
        if st.session_state.coords:
            path = None
            if st.session_state.route_result:
                path = st.session_state.route_result.get('path')
            
            fig = plot_graph_with_route(st.session_state.coords, path)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Display results
        if st.session_state.route_result:
            result = st.session_state.route_result
            
            st.subheader("📈 Route Results")
            
            # Metrics
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Algorithm", result.get('algorithm', 'GA').upper())
            with metric_cols[1]:
                st.metric("Estimated Time", f"{result['estimated_time_min']:.2f} min")
            with metric_cols[2]:
                st.metric("Path Length", len(result['path']) if result['path'] else 0)
            
            # Path details
            if result.get('path'):
                st.write("**Route Path:**")
                st.code(" → ".join(map(str, result['path'])))
# Add this function
def check_backend():
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Add this check before showing UI
if not check_backend():
    st.error("⚠️ Backend not running! Start with: python app.py")
    st.stop()
    
# Footer
st.divider()
st.markdown("""
### 🔬 Technical Details

**Machine Learning Model:**
- Architecture: 3-layer ANN with dropout regularization
- Input features: hour, day, holiday status, base speed, distance, base congestion
- Output: 3-class congestion prediction (Low/Medium/High)

**Optimization Algorithms:**
- **Dijkstra**: O(E log V) time complexity for weighted shortest path
- **A***: Euclidean distance heuristic for faster convergence
- **Genetic Algorithm**: Population-based evolutionary optimization with tournament selection
""")