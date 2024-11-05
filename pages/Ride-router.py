import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
import numpy as np
from geopy.distance import geodesic
from collections import defaultdict
import plotly.express as px
import math
import json
from datetime import datetime
def load_css():
    # External CSS dependencies
    st.markdown(
        """
        <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        """,
        unsafe_allow_html=True
    )

    # Custom CSS to hide Streamlit components and adjust layout
    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .main {
                margin-top: -20px;
                padding-top: 10px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .navbar {
                padding: 1rem;
                margin-bottom: 2rem;
            }
            .card {
                padding: 1rem;
                margin-bottom: 1rem;
                transition: transform 0.2s;
                border-radius:5px;
            }
            .card:hover {
                transform: scale(1.02);
            }
            .navbar-brand img {
                margin-right: 10px;
                height: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def create_navbar():
    st.markdown(
        """
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #4267B2;">
            <a class="navbar-brand" href="#" target="_blank">
                <img src="https://www.4th-ir.com/favicon.ico" alt="4th-ir logo">
                Ride Router
            </a>
        </nav>
        """,
        unsafe_allow_html=True
    )


class StaffTransportOptimizer:
    def __init__(self):
        self.office_location = {
            'lat': 5.582636441579255,
            'lon': -0.143551646497661
        }
        self.MAX_PASSENGERS = 4
        
    def load_sample_data(self):
        """Load sample staff location data for Accra region"""
        return pd.DataFrame({
            'staff_id': range(1, 21),
            'name': [f'Employee {i}' for i in range(1, 21)],
            'latitude': np.random.uniform(5.5526, 5.6126, 20),
            'longitude': np.random.uniform(-0.1735, -0.1135, 20),
            'address': [
                'Adabraka', 'Osu', 'Cantonments', 'Airport Residential',
                'East Legon', 'Spintex', 'Tema', 'Teshie', 'Labadi',
                'Labone', 'Ridge', 'Roman Ridge', 'Dzorwulu', 'Abelemkpe',
                'North Kaneshie', 'Dansoman', 'Mamprobi', 'Chorkor',
                'Abeka', 'Achimota'
            ]
        })

    def create_clusters(self, staff_data, eps_km=2):
        """Create clusters based on staff locations"""
        if staff_data is None or len(staff_data) == 0:
            return None
        
        eps_degrees = eps_km / 111
        coords = staff_data[['latitude', 'longitude']].values
        db = DBSCAN(eps=eps_degrees, min_samples=1).fit(coords)
        staff_data['cluster'] = db.labels_
        return staff_data

    def optimize_routes(self, staff_data):
        """Optimize routes for each cluster"""
        routes = defaultdict(list)
        
        for cluster_id in staff_data['cluster'].unique():
            cluster_group = staff_data[staff_data['cluster'] == cluster_id].copy()
            
            cluster_group['distance_to_office'] = cluster_group.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            sorted_group = cluster_group.sort_values('distance_to_office', ascending=False)
            
            for i in range(0, len(sorted_group), self.MAX_PASSENGERS):
                car_group = sorted_group.iloc[i:i + self.MAX_PASSENGERS]
                routes[f'Cluster {cluster_id} - Car {i//self.MAX_PASSENGERS + 1}'] = car_group.to_dict('records')
                
        return routes

    def create_map(self, routes):
        """Create an interactive map with dashed route lines"""
        m = folium.Map(
            location=[self.office_location['lat'], self.office_location['lon']],
            zoom_start=13,
            tiles="cartodbpositron"
        )
        
        # Add office marker
        folium.Marker(
            [self.office_location['lat'], self.office_location['lon']],
            popup='Office',
            icon=folium.Icon(color='red', icon='building', prefix='fa'),
            tooltip="Office Location"
        ).add_to(m)
        
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
        
        for route_idx, (route_name, group) in enumerate(routes.items()):
            color = colors[route_idx % len(colors)]
            route_group = folium.FeatureGroup(name=route_name)
            
            # Create coordinates list for the route
            coordinates = [[staff['latitude'], staff['longitude']] for staff in group]
            coordinates.append([self.office_location['lat'], self.office_location['lon']])
            
            # Add dashed line for route
            folium.PolyLine(
                coordinates,
                weight=2,
                color=color,
                opacity=0.6,
                dash_array='5, 10',  # Create dashed line effect
                popup=route_name
            ).add_to(route_group)
            
            # Add staff markers
            for staff in group:
                folium.CircleMarker(
                    [staff['latitude'], staff['longitude']],
                    radius=6,
                    popup=f"""
                    <b>{staff['name']}</b><br>
                    Address: {staff['address']}<br>
                    Distance to office: {staff['distance_to_office']:.2f} km
                    """,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=staff['name']
                ).add_to(route_group)
            
            route_group.add_to(m)
        
        folium.LayerControl().add_to(m)
        return m

def init_session_state():
    """Initialize session state variables"""
    if 'staff_data' not in st.session_state:
        st.session_state.staff_data = None
    if 'routes' not in st.session_state:
        st.session_state.routes = None
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    if 'saved_sessions' not in st.session_state:
        st.session_state.saved_sessions = {}

def save_current_session():
    """Save current session data"""
    if st.session_state.staff_data is not None and st.session_state.routes is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_data = {
            'staff_data': st.session_state.staff_data.to_dict('records'),
            'routes': st.session_state.routes
        }
        st.session_state.saved_sessions[timestamp] = session_data
        return timestamp
    return None

def load_session(timestamp):
    """Load saved session data"""
    if timestamp in st.session_state.saved_sessions:
        session_data = st.session_state.saved_sessions[timestamp]
        st.session_state.staff_data = pd.DataFrame(session_data['staff_data'])
        st.session_state.routes = session_data['routes']
        st.session_state.optimization_done = True

def main():
    load_css()
    create_navbar()
    
    init_session_state()
    optimizer = StaffTransportOptimizer()
    
    # Sidebar controls
    with st.sidebar:
        st.title("Controls")
        
        # Data input section
        st.header("Data Input")
        data_option = st.radio(
            "Choose data input method",
            ["Use Sample Data", "Upload CSV"]
        )
        
        if data_option == "Use Sample Data":
            if st.button("Load Sample Data"):
                st.session_state.staff_data = optimizer.load_sample_data()
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type="csv"
            )
            if uploaded_file is not None:
                st.session_state.staff_data = pd.read_csv(uploaded_file)
        
        # Optimization parameters
        if st.session_state.staff_data is not None:
            st.header("Optimization Parameters")
            eps_km = st.slider(
                "Cluster radius (km)",
                0.5, 5.0, 2.0, 0.1
            )
            
            if st.button("Optimize Routes", type="primary"):
                clustered_data = optimizer.create_clusters(st.session_state.staff_data, eps_km)
                st.session_state.routes = optimizer.optimize_routes(clustered_data)
                st.session_state.optimization_done = True
        
        # Session management
        st.header("Session Management")
        if st.button("Save Current Session"):
            timestamp = save_current_session()
            if timestamp:
                st.success(f"Session saved at {timestamp}")
        
        if st.session_state.saved_sessions:
            selected_session = st.selectbox(
                "Load saved session",
                options=list(st.session_state.saved_sessions.keys()),
                format_func=lambda x: f"Session from {x}"
            )
            if st.button("Load Selected Session"):
                load_session(selected_session)
                st.success("Session loaded successfully")

    # Main content area
    st.markdown("<div class='alert alert-success'><marquee> Staff Transportation Optimizer - Accra Office</marquee></div>",unsafe_allow_html=True)
    
    # Display current data and results
    if st.session_state.staff_data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.optimization_done:
                st.header("Route Map")
                m = optimizer.create_map(st.session_state.routes)
                st_folium(m, width=None, height=600)
        
        with col2:
            st.header("Staff Data")
            st.dataframe(
                st.session_state.staff_data,
                hide_index=True,
                height=300
            )
            
            if st.session_state.optimization_done:
                st.header("Route Details")
                for route_name, group in st.session_state.routes.items():
                    with st.expander(route_name):
                        route_df = pd.DataFrame(group)
                        st.dataframe(
                            route_df[['name', 'address', 'distance_to_office']],
                            hide_index=True
                        )

if __name__ == "__main__":
    main()