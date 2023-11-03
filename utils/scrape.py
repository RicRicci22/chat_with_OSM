'''
This module contains functions to scrape data from openstreetmap.
'''

import requests
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List
from PIL import Image
from io import BytesIO

def fetch_overpass_data(bbox):
    """
    Fetch OSM data within a bounding box using the Overpass API.
    
    Parameters:
    bbox: (bottom, left, top, right) tuple of coordinates in decimal degrees of the bounding box
    
    Returns:
    str: OSM data as a string in JSON format.
    """
    
    bottom, left, top, right = bbox
    
    # Define the Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Define the Overpass QL query
    overpass_query = f"""
    [out:json];
    (
      node({bottom},{left},{top},{right});
      way({bottom},{left},{top},{right});
    );
    out body;
    """
    # relation({bottom},{left},{top},{right});
    # Make the API request
    response = requests.get(overpass_url, params={'data': overpass_query})
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        return f"Failed to fetch data: {response.status_code}"
    
def get_rbg_image(bbox):
    '''
    This function returns a RGB image from mapbox satellite-v9 style. 
    Input:
        bbox: (bottom, left, top, right) tuple of coordinates in decimal degrees of the bounding box
    Output:
        image: RGB image
    '''
    
    bottom, left, top, right = bbox
    
    # Your Mapbox Access Token
    ACCESS_TOKEN = 'pk.eyJ1IjoicmljY2FyZG85Njk2IiwiYSI6ImNsb2NudnRrZzBuNmkycW42ankxdzlnaHEifQ.ZQ02zwQEeGEWWjvLHgIibQ'
    
    # Fix the
    zoom = 18
    
    # Calculate the width and height of the bounding box in meters
    width_bbox, height_bbox = calculate_bbox_dimensions(bbox)
    # Calculate the pixel size in meters given the predefined zoom and the latitude
    pixsize = get_pixel_dimension_in_meters(zoom, (bottom + top) / 2)
    
    # Get the dimensions of the bbox 
    width, height = int(width_bbox / pixsize), int(height_bbox / pixsize)
    
    center_lon, center_lat = (left + right) / 2, (bottom + top) / 2

    # Create the URL
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{center_lon},{center_lat},{zoom}/{width}x{height}?access_token={ACCESS_TOKEN}"

    # Fetch the image
    response = requests.get(url)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    else:
        print(f"Failed to get image: {response.content}")
        return None
        
        
def calculate_bbox_dimensions(bbox):
    """
    Calculates the width and height in meters of a geographic bounding box (bbox) on Earth's surface.
    
    This function employs the Haversine formula to approximate the distance between two points
    on a sphere given their longitudes and latitudes. It considers the Earth's curvature and provides
    the width and height of the bbox in meters, assuming Earth is a perfect sphere.
    
    Parameters:
    bbox : tuple of float
        A tuple containing the decimal degrees of the bbox in the format (bottom, left, top, right),
        where:
        - 'bottom' is the southernmost latitude,
        - 'left' is the westernmost longitude,
        - 'top' is the northernmost latitude,
        - 'right' is the easternmost longitude.

    Returns:
    (float, float)
        A tuple containing the width (east-west length) and height (north-south length) of the bbox in meters.

    Example:
    >>> calculate_bbox_dimensions((34.0, -118.0, 34.1, -117.9))
    (11131.949079327344, 11119.555392036573)
    """
    
    bottom, left, top, right = bbox
    
    R = 6371e3  # Earth radius in meters
    
    # Convert degrees to radians
    bottom = radians(bottom)
    top = radians(top)
    left = radians(left)
    right = radians(right)

    # Haversine formula for width (along a line of latitude)
    dlon = right - left
    a_width = sin(dlon/2) * sin(dlon/2) * cos(bottom) * cos(top)
    c_width = 2 * atan2(sqrt(a_width), sqrt(1-a_width))
    width_meters = R * c_width

    # Haversine formula for height (along a line of longitude)
    dlat = top - bottom
    a_height = sin(dlat/2) * sin(dlat/2)
    c_height = 2 * atan2(sqrt(a_height), sqrt(1-a_height))
    height_meters = R * c_height
    
    return width_meters, height_meters

def get_pixel_dimension_in_meters(zoom, latitude):
    pixel_dimensions = {
        0: {0: 78271.484, 20: 73551.136, 40: 59959.436, 60: 39135.742, 80: 13591.701},
        1: {0: 39135.742, 20: 36775.568, 40: 29979.718, 60: 19567.871, 80: 6795.850},
        2: {0: 19567.871, 20: 18387.784, 40: 14989.859, 60: 9783.936, 80: 3397.925},
        3: {0: 9783.936, 20: 9193.892, 40: 7494.929, 60: 4891.968, 80: 1698.963},
        4: {0: 4891.968, 20: 4596.946, 40: 3747.465, 60: 2445.984, 80: 849.481},
        5: {0: 2445.984, 20: 2298.473, 40: 1873.732, 60: 1222.992, 80: 424.741},
        6: {0: 1222.992, 20: 1149.237, 40: 936.866, 60: 611.496, 80: 212.370},
        7: {0: 611.496, 20: 574.618, 40: 468.433, 60: 305.748, 80: 106.185},
        8: {0: 305.748, 20: 287.309, 40: 234.217, 60: 152.874, 80: 53.093},
        9: {0: 152.874, 20: 143.655, 40: 117.108, 60: 76.437, 80: 26.546},
        10: {0: 76.437, 20: 71.827, 40: 58.554, 60: 38.218, 80: 13.273},
        11: {0: 38.218, 20: 35.914, 40: 29.277, 60: 19.109, 80: 6.637},
        12: {0: 19.109, 20: 17.957, 40: 14.639, 60: 9.555, 80: 3.318},
        13: {0: 9.555, 20: 8.978, 40: 7.319, 60: 4.777, 80: 1.659},
        14: {0: 4.777, 20: 4.489, 40: 3.660, 60: 2.389, 80: 0.830},
        15: {0: 2.389, 20: 2.245, 40: 1.830, 60: 1.194, 80: 0.415},
        16: {0: 1.194, 20: 1.122, 40: 0.915, 60: 0.597, 80: 0.207},
        17: {0: 0.597, 20: 0.561, 40: 0.457, 60: 0.299, 80: 0.104},
        18: {0: 0.299, 20: 0.281, 40: 0.229, 60: 0.149, 80: 0.052},
        19: {0: 0.149, 20: 0.140, 40: 0.114, 60: 0.075, 80: 0.026},
        20: {0: 0.075, 20: 0.070, 40: 0.057, 60: 0.037, 80: 0.013},
        21: {0: 0.037, 20: 0.035, 40: 0.029, 60: 0.019, 80: 0.006},
        22: {0: 0.019, 20: 0.018, 40: 0.014, 60: 0.009, 80: 0.003}
    }

    # Find the closest latitude in the available data
    closest_latitude = min(pixel_dimensions[zoom].keys(), key=lambda x: abs(x - latitude))
    
    return pixel_dimensions[zoom][closest_latitude]


def filter_osm_data(osm_data: List, elements_to_keep: List = []):
    """
    Filters out latitude and longitude coordinates from OpenStreetMap (OSM) data elements.
    For elements of the type 'way', includes only the name tag.

    Parameters:
    osm_data (Dict): The original OSM data in dictionary form.
                     The dictionary must contain a key 'elements' that points to a list of dictionaries.
                     Each dictionary in the list represents an OSM element and may contain the keys: 'type', 'id', 'lat', 'lon', 'tags', 'nodes', 'members'.
    elements_to_keep (List): A list of strings representing the keys of the tags to keep. If the list is empty, all tags are kept.

    Returns:
    Dict: A new dictionary containing filtered elements. The 'lat' and 'lon' keys are removed from each element.
          For 'way' elements, only the 'name' tag is included, and the 'nodes' key is removed.
          
    """
    filtered_elements=[]
    for element in osm_data:
        # Keep if at least one of the tags is present in elements_to_keep 
        for key in element.keys():
            if key in elements_to_keep:
                filtered_elements.append(element)
                break
            
    return filtered_elements

def get_center_way(nodes_id, nodes, debug=False):
    lats = []
    longs = []
    for node_id in nodes_id:
        try:
            lats.append(nodes[node_id][0])
            longs.append(nodes[node_id][1])
        except KeyError:
            if(debug):
                print("Presumably way that extends outside the bbox")
            
    if len(lats) == 0 or len(longs) == 0:
        return None, None
    else:
        # Get the center
        center_lat = sum(lats) / len(lats)
        center_lon = sum(longs) / len(longs)
        
        return center_lat, center_lon
        


def proj_lat_lon_on_image(bbox, osm_data, nodes):
    '''
    This function takes a list of osm elements and project the location based on the bounding box. 
    It divides the image in 9 portions, denominated as 
    |-----------------|-----------------|-----------------|
    |                 |                 |                 |
    |   Top Left      |   Top Center    |   Top Right     |
    |                 |                 |                 |
    |-----------------|-----------------|-----------------|
    |                 |                 |                 |
    |   Center Left   |   Center        |   Center Right  |
    |                 |                 |                 |
    |-----------------|-----------------|-----------------|
    |                 |                 |                 |
    |   Bottom Left   |   Bottom Center |   Bottom Right  |
    |                 |                 |                 |
    |-----------------|-----------------|-----------------|
    '''
    bottom, left, top, right = bbox
    # Calculate the increments for latitude and longitude to split into three equal parts
    lat_increment = (top - bottom) / 3
    lon_increment = (right - left) / 3

    # Calculate the starting points for latitude and longitude
    start_lat = bottom + lat_increment / 2
    start_lon = left + lon_increment / 2

    # List to hold the center of each tile
    tiles_centers = [
        (start_lat + i * lat_increment, start_lon + j * lon_increment)
        for i in range(3) for j in range(3)
    ]
    placing_identifiers = ["bottom left", "bottom center", "bottom right", "center left", "center", "center right", "top left", "top center", "top right"]
    
    # Convert each element position in osm_data in an identifier, based on the closest tile center
    # Approximate using the euclidean distance on latitude and longitude coordinates
    located_elements = []
    for element in osm_data['elements']:
        assert "tags" in element.keys(), "Error in the prior filtering of nodes!"
        if element["type"]=="node":
            lat, lon = element["lat"], element["lon"]
        elif element["type"]=="way":
            # First get the center
            lat, lon = get_center_way(element["nodes"], nodes)
        else:
            raise ValueError("Element type not yet supported!")
        
        # Get the closest center 
        if lat != None and lon != None:
            min_dist = float("inf")
            for i, center in enumerate(tiles_centers):
                dist = sqrt((lat - center[0])**2 + (lon - center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    element["tags"]["position"] = placing_identifiers[i]
        
            located_elements.append(element["tags"])
    
    return located_elements
        
def get_isolated_nodes(osm_data):
    '''
    This function returns a list of nodes that are not part of any way. 
    '''
    nodes = {}
    filtered = {"elements":[]}
    for element in osm_data['elements']:
        if element['type'] == "node" and "tags" not in element.keys():
            nodes[element['id']] = (element['lat'], element['lon'])
        elif "tags" in element.keys():
            filtered["elements"].append(element)
            
    return nodes, filtered

if __name__ == "__main__":
    #     /*
    # This query looks for nodes, ways and relations 
    # with the given key/value combination.
    # Choose your region and hit the Run button above!
    # */
    # [out:json][timeout:25];
    # // gather results
    # (
    #   // query part for: “amenity=post_box”
    #   way(46.044825285675955,10.983120203018188,46.04676439737953,10.986515879631042);
    # );
    # // print results
    # out body;
    # >;
    # out skel qt;
    
    # Test the function with a sample bounding box
    #left, bottom, right, top = -0.130, 51.507, -0.129, 51.508  # Around the area of Big Ben in London
    bottom, left, top, right = 46.04496229703382,10.98408579826355,46.04642930052508,10.986006259918213  # Parco Nadac, in Calavino, my place :)
    bbox = (bottom, left, top, right)
    osm_data = fetch_overpass_data(bbox)
    nodes, filtered = get_isolated_nodes(osm_data)
    located_elements = proj_lat_lon_on_image(bbox, filtered, nodes)
    for element in located_elements:
        print(element)
    # image = get_rbg_image(bottom, left, top, right)
    # # print(type(osm_data))
    # # print(osm_data.keys())
    # # print(type(osm_data['elements']))
    # # print(len(osm_data['elements']))
    # # # Print the first element that is an amenity
    # # print(osm_data['elements'])
    # #print(osm_data['elements'][2])
    # lol = filter_osm_data(osm_data)

    # buildings = 0 
    # for element in lol['elements']:
    #     if "building" in element.keys():
    #         buildings += 1
    
    # print(buildings)