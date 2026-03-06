"""
Road Network Graph Extraction Module
Converts binary road masks into navigable graph structures
"""
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
from scipy import ndimage
import networkx as nx
import json


def skeletonize_mask(mask):
    """
    Reduce binary mask to 1-pixel-wide skeleton
    
    Args:
        mask: Binary mask as numpy array (H, W) with values 0 or 255
    
    Returns:
        numpy array: Skeleton as binary image (H, W)
    """
    # Normalize mask to binary (0 or 1)
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Apply skeletonization
    skeleton = skeletonize(binary_mask).astype(np.uint8)
    
    return skeleton


def get_neighbor_count(skeleton, y, x):
    """
    Count the number of 8-connected neighbors for a pixel
    
    Args:
        skeleton: Binary skeleton array
        y, x: Pixel coordinates
    
    Returns:
        int: Number of neighbors (0-8)
    """
    h, w = skeleton.shape
    count = 0
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx_coord = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx_coord < w:
                if skeleton[ny, nx_coord] > 0:
                    count += 1
    
    return count


def find_nodes(skeleton):
    """
    Detect intersections (3+ neighbors) and endpoints (1 neighbor)
    
    Args:
        skeleton: Binary skeleton as numpy array
    
    Returns:
        dict: {
            'intersections': [(y, x), ...],
            'endpoints': [(y, x), ...],
            'all_nodes': [(y, x, type), ...]
        }
    """
    intersections = 
    endpoints = 
    all_nodes = 
    
    # Find all skeleton pixels
    skeleton_points = np.argwhere(skeleton > 0)
    
    for point in skeleton_points:
        y, x = point
        neighbors = get_neighbor_count(skeleton, y, x)
        
        if neighbors == 1:
            # Endpoint
            endpoints.append((int(y), int(x)))
            all_nodes.append((int(y), int(x), 'endpoint'))
        elif neighbors >= 3:
            # Intersection
            intersections.append((int(y), int(x)))
            all_nodes.append((int(y), int(x), 'intersection'))
    
    return {
        'intersections': intersections,
        'endpoints': endpoints,
        'all_nodes': all_nodes
    }


def trace_edge(skeleton, start_y, start_x, visited_global):
    """
    Trace a single edge from a starting point until hitting a node or dead end
    
    Args:
        skeleton: Binary skeleton
        start_y, start_x: Starting coordinates
        visited_global: Global visited set
    
    Returns:
        list: List of (y, x) coordinates forming the edge path
    """
    path = [(start_y, start_x)]
    visited_local = {(start_y, start_x)}
    current_y, current_x = start_y, start_x
    
    while True:
        neighbors = get_neighbor_count(skeleton, current_y, current_x)
        
        # If we hit an intersection or endpoint (after moving), stop
        if len(path) > 1 and (neighbors >= 3 or neighbors == 1):
            break
        
        # Find next unvisited neighbor
        found_next = False
        h, w = skeleton.shape
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_coord = current_y + dy, current_x + dx
                
                if 0 <= ny < h and 0 <= nx_coord < w:
                    if skeleton[ny, nx_coord] > 0 and (ny, nx_coord) not in visited_local:
                        path.append((ny, nx_coord))
                        visited_local.add((ny, nx_coord))
                        current_y, current_x = ny, nx_coord
                        found_next = True
                        break
            if found_next:
                break
        
        if not found_next:
            break
    
    return path


def extract_edges(skeleton, nodes_dict):
    """
    Extract road segments (edges) between nodes
    
    Args:
        skeleton: Binary skeleton
        nodes_dict: Output from find_nodes()
    
    Returns:
        list: List of edge dictionaries with 'start', 'end', 'path', 'length'
    """
    edges = 
    all_nodes = set((n[0], n[1]) for n in nodes_dict['all_nodes'])
    visited_edges = set()
    
    # For each node, trace outgoing edges
    for node in nodes_dict['all_nodes']:
        node_y, node_x, node_type = node
        
        # Find all neighbors of this node
        h, w = skeleton.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                    
                ny, nx_coord = node_y + dy, node_x + dx
                if 0 <= ny < h and 0 <= nx_coord < w:
                    if skeleton[ny, nx_coord] > 0:
                        # Trace this edge
                        path = trace_edge(skeleton, ny, nx_coord, set())
                        
                        if len(path) >= 1:
                            # Find the end node
                            end_y, end_x = path[-1]
                            
                            # Check if end point is near a node
                            end_node = None
                            for n in nodes_dict['all_nodes']:
                                if abs(n[0] - end_y) <= 1 and abs(n[1] - end_x) <= 1:
                                    end_node = (n[0], n[1])
                                    break
                            
                            if end_node is None:
                                end_node = (end_y, end_x)
                            
                            # Create edge key (sorted to avoid duplicates)
                            start_node = (node_y, node_x)
                            edge_key = tuple(sorted([start_node, end_node]))
                            
                            if edge_key not in visited_edges:
                                visited_edges.add(edge_key)
                                
                                # Calculate edge length (Euclidean path length)
                                full_path = [(node_y, node_x)] + path
                                length = 0
                                for i in range(1, len(full_path)):
                                    dy = full_path[i][0] - full_path[i-1][0]
                                    dx = full_path[i][1] - full_path[i-1][1]
                                    length += np.sqrt(dy*dy + dx*dx)
                                
                                edges.append({
                                    'start': start_node,
                                    'end': end_node,
                                    'path': full_path,
                                    'length': float(length)
                                })
    
    return edges


def build_graph(nodes_dict, edges):
    """
    Create NetworkX graph from nodes and edges
    
    Args:
        nodes_dict: Output from find_nodes()
        edges: Output from extract_edges()
    
    Returns:
        networkx.Graph: Road network graph
    """
    G = nx.Graph()
    
    # Add nodes
    for node in nodes_dict['all_nodes']:
        y, x, node_type = node
        G.add_node((y, x), pos=(x, y), type=node_type)
    
    # Add edges
    for edge in edges:
        G.add_edge(
            edge['start'], 
            edge['end'], 
            length=edge['length'],
            path=edge['path']
        )
    
    return G


def calculate_statistics(graph, edges, skeleton):
    """
    Calculate network statistics
    
    Args:
        graph: NetworkX graph
        edges: List of edges
        skeleton: Binary skeleton
    
    Returns:
        dict: Statistics dictionary
    """
    # Basic counts
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    # Total road length in pixels
    total_length = sum(e['length'] for e in edges)
    
    # Count node types
    num_intersections = sum(1 for _, data in graph.nodes(data=True) if data.get('type') == 'intersection')
    num_endpoints = sum(1 for _, data in graph.nodes(data=True) if data.get('type') == 'endpoint')
    
    # Dead ends (endpoint count)
    dead_ends = num_endpoints
    
    # Connected components
    num_components = nx.number_connected_components(graph)
    
    # Average degree
    if num_nodes > 0:
        avg_degree = sum(dict(graph.degree()).values()) / num_nodes
    else:
        avg_degree = 0
    
    # Skeleton pixel count
    skeleton_pixels = int(np.sum(skeleton > 0))
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_intersections': num_intersections,
        'num_endpoints': num_endpoints,
        'dead_ends': dead_ends,
        'total_length_pixels': round(total_length, 2),
        'num_connected_components': num_components,
        'average_degree': round(avg_degree, 2),
        'skeleton_pixels': skeleton_pixels
    }


def export_geojson(graph, edges):
    """
    Convert graph to GeoJSON format
    
    Note: Coordinates are in pixel space (y, x).
    For real-world coordinates, georeferencing would be needed.
    
    Args:
        graph: NetworkX graph
        edges: List of edges with paths
    
    Returns:
        dict: GeoJSON FeatureCollection
    """
    features = 
    
    # Add nodes as Point features
    for node, data in graph.nodes(data=True):
        y, x = node
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x, y]  # GeoJSON uses [lon, lat] or [x, y]
            },
            "properties": {
                "type": data.get('type', 'unknown'),
                "id": f"node_{y}_{x}"
            }
        }
        features.append(feature)
    
    # Add edges as LineString features
    for i, edge in enumerate(edges):
        # Convert path to coordinates
        coordinates = [[p[1], p[0]] for p in edge['path']]  # Convert (y,x) to [x,y]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "id": f"edge_{i}",
                "length_pixels": edge['length'],
                "start": f"{edge['start'][0]}_{edge['start'][1]}",
                "end": f"{edge['end'][0]}_{edge['end'][1]}"
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson


def extract_road_network(mask):
    """
    Main function: Extract complete road network from binary mask
    
    Args:
        mask: Binary road mask as numpy array (H, W) with values 0/255
    
    Returns:
        dict: {
            'skeleton': skeleton array,
            'nodes': nodes dictionary,
            'edges': list of edges,
            'graph': NetworkX graph,
            'statistics': network stats,
            'geojson': GeoJSON FeatureCollection
        }
    """
    # Step 1: Skeletonize
    skeleton = skeletonize_mask(mask)
    
    # Step 2: Find nodes
    nodes = find_nodes(skeleton)
    
    # Step 3: Extract edges
    edges = extract_edges(skeleton, nodes)
    
    # Step 4: Build graph
    graph = build_graph(nodes, edges)
    
    # Step 5: Calculate statistics
    statistics = calculate_statistics(graph, edges, skeleton)
    
    # Step 6: Export to GeoJSON
    geojson = export_geojson(graph, edges)
    
    return {
        'skeleton': skeleton,
        'nodes': nodes,
        'edges': edges,
        'graph': graph,
        'statistics': statistics,
        'geojson': geojson
    }


def create_skeleton_overlay(original_image, skeleton, nodes_dict, 
                            skeleton_color=(0, 255, 0), 
                            intersection_color=(255, 0, 0),
                            endpoint_color=(0, 0, 255),
                            node_radius=5):
    """
    Create visualization overlay with skeleton and nodes
    
    Args:
        original_image: Original image as numpy array (H, W, 3)
        skeleton: Binary skeleton
        nodes_dict: Nodes dictionary from find_nodes()
        skeleton_color: RGB color for skeleton lines
        intersection_color: RGB color for intersection nodes
        endpoint_color: RGB color for endpoint nodes
        node_radius: Radius of node circles
    
    Returns:
        numpy array: Overlay image (H, W, 3)
    """
    # Create copy of original
    overlay = original_image.copy()
    
    # Draw skeleton in green
    skeleton_mask = skeleton > 0
    for c in range(3):
        overlay[:, :, c] = np.where(skeleton_mask, skeleton_color[c], overlay[:, :, c])
    
    # Draw nodes as circles
    h, w = skeleton.shape
    
    # Draw intersections (red circles)
    for y, x in nodes_dict['intersections']:
        for dy in range(-node_radius, node_radius + 1):
            for dx in range(-node_radius, node_radius + 1):
                if dy*dy + dx*dx <= node_radius*node_radius:
                    ny, nx_coord = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx_coord < w:
                        overlay[ny, nx_coord] = intersection_color
    
    # Draw endpoints (blue circles)
    for y, x in nodes_dict['endpoints']:
        for dy in range(-node_radius, node_radius + 1):
            for dx in range(-node_radius, node_radius + 1):
                if dy*dy + dx*dx <= node_radius*node_radius:
                    ny, nx_coord = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx_coord < w:
                        overlay[ny, nx_coord] = endpoint_color
    
    return overlay.astype(np.uint8)
