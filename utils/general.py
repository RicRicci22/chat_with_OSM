def get_bbox_bltr(bbox_coords):
    '''
    This function returns the bbox in the format [bottom, left, top, right]
    '''
    latitudes = [x[1] for x in bbox_coords[0]]
    longitudes = [x[0] for x in bbox_coords[0]]
    bottom = min(latitudes)
    top = max(latitudes)
    left = min(longitudes)
    right = max(longitudes)
    
    return [bottom, left, top, right]