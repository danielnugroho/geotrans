# -*- coding: utf-8 -*-

__version__ = "1.0.6"
__author__ = "Daniel Adi Nugroho"
__email__ = "dnugroho@gmail.com"
__status__ = "Production"
__date__ = "2025-02-10"
__copyright__ = "Copyright (c) 2025"
__license__ = "MIT"  # or appropriate license

# Version History
# --------------

# 1.0.6 (2025-02-11)
# - combined three subsections in transform_geotiff for multiprocessing
# - move the band data reading section to outside blocks loop

# 1.0.5 (2025-02-11)
# - started to employ multiprocessing on world coord trf successfully

# 1.0.4 (2025-02-10)
# - added timing commentary on TIF transform to analyze optimization

# 1.0.3 (2025-02-08)
# - fixed the NoData values problem (NoData being interpolated and damage the output)

# 1.0.2 (2025-02-08)
# - GeoTIFF transform is functional adequately FAST! Thanks to vectorized ops.
# - GeoTIfF transfrom is spread into chunks to reduce memory footprint
# - there is still problem in GeoTIFF transform due to NoData values
# - Further speed optimization through parallel processing is possible

# 1.0.1 (2025-02-08)
# - GeoTIFF functionality is still fixed, but still slow.

# 1.0.0 (2025-02-08)
# - Initial release
# - Support for 2D and 3D transformations
# - Helmert and Affine transformation options
# - Good CSV, LAZ, and DXF functionality
# - GeoTIFF functionality is still broken

"""
Geospatial Data Transformation Module
==============================

This script provides functionality for transforming spatial data between different coordinate systems
using either Helmert or Affine transformations in both 2D and 3D space. It supports multiple file 
formats including CSV, LAZ/LAS point clouds, DXF CAD files, and GeoTIFF raster data.

Purpose:
--------
- Transform spatial data between different coordinate systems
- Support both 2D and 3D coordinate transformations
- Handle multiple file formats (CSV, LAZ/LAS, DXF, GeoTIFF)
- Provide both Helmert (similarity) and Affine transformation options
- Preserve data integrity during transformation

Requirements:
------------
- Python 3.6 or higher
- Required packages: 
  - numpy: for numerical computations
  - pdal: for point cloud processing
  - ezdxf: for CAD file handling
  - rasterio: for GeoTIFF processing
  - pathlib: for path handling

Input Formats:
-------------
1. CSV files:
   - Must contain columns: ID, X, Y, Z
   - Optional column: Type

2. LAZ/LAS files:
   - Standard LAS/LAZ point cloud format
   - Maintains all point attributes

3. DXF files:
   - Supports various entity types (points, lines, polylines, etc.)
   - Preserves layers and other CAD properties

4. GeoTIFF files:
   - Maintains georeferencing information
   - Preserves all bands and metadata

Usage:
------
python transform_coords.py input_file output_file transform_params.txt

Parameter File Format:
--------------------
Required parameters depend on transformation type:
1. Helmert (Similarity):
   - 2D: mode=2D, type=helmert, cx, cy, rotation, scale, tx, ty
   - 3D: mode=3D, type=helmert, cx, cy, cz, omega, phi, kappa, scale, tx, ty, tz

2. Affine:
   - 2D: mode=2D, type=affine, cx, cy, a11, a12, a21, a22, tx, ty
   - 3D: mode=3D, type=affine, cx, cy, cz, a11...a33, tx, ty, tz

Output:
-------
- Transformed spatial data in the same format as input
- Maintains original file structure and properties
- Preserves metadata and attributes

Notes:
------
- Input and output files must be of the same type
- All transformations are applied relative to specified center point
- Error handling includes detailed messages for troubleshooting
- Progress feedback provided during transformation
"""

import numpy as np
import pdal
import json
import csv
import sys
import os
import ray
from pathlib import Path
import ezdxf
from ezdxf.math import Vec3
import rasterio
from datetime import datetime
from scipy.ndimage import map_coordinates

from rasterio.transform import from_origin
from rasterio.windows import Window
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


# [Previous functions remain the same until transform_dxf]
# Include all the previous functions for CSV, LAZ, and DXF processing

def read_params(param_file):
    """Read transformation parameters from file"""
    params = {}
    try:
        with open(param_file, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                # Convert numeric values to float
                if key not in ['mode', 'type']:
                    value = float(value)
                params[key] = value
        return params
    except Exception as e:
        print(f"Error reading parameter file: {str(e)}")
        sys.exit(1)

def read_csv_coordinates(coord_file):
    """Read coordinates from CSV file"""
    try:
        with open(coord_file, 'r') as file:
            csv_reader = csv.reader(file)
            
            # Initialize lists for data
            ids = []
            coords = []
            types = []
            
            for row in csv_reader:
                try:
                    ids.append(row[0])
                    coords.append([float(row[1]), float(row[2]), float(row[3])])
                    if len(row) > 4:  # If type column exists
                        types.append(row[4])
                    else:
                        types.append('')
                except ValueError as e:
                    print(f"Error parsing row {row}: {str(e)}")
                    continue
                    
            return np.array(coords), ids, types
            
    except Exception as e:
        print(f"Error reading coordinate file: {str(e)}")
        sys.exit(1)

def save_csv_coordinates(coords, ids, types, output_file):
    """Save transformed coordinates to CSV file"""
    try:
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write coordinates
            for i in range(len(coords)):
                writer.writerow([
                    ids[i],
                    f"{coords[i][0]:.3f}",
                    f"{coords[i][1]:.3f}",
                    f"{coords[i][2]:.3f}",
                    types[i]
                ])
                
    except Exception as e:
        print(f"Error saving coordinates: {str(e)}")
        sys.exit(1)

def create_2d_helmert_matrix(theta):
    """Create 2D Helmert rotation matrix"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

def create_2d_affine_matrix(a11, a12, a21, a22):
    """Create 2D affine transformation matrix"""
    return np.array([[a11, a12],
                    [a21, a22]])

def create_3d_helmert_matrix(omega, phi, kappa):
    """Create 3D Helmert rotation matrix"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(omega), -np.sin(omega)],
                   [0, np.sin(omega), np.cos(omega)]])
    
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])
    
    Rz = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                   [np.sin(kappa), np.cos(kappa), 0],
                   [0, 0, 1]])
    
    return Rx @ Ry @ Rz

def create_3d_affine_matrix(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    """Create 3D affine transformation matrix"""
    return np.array([[a11, a12, a13],
                    [a21, a22, a23],
                    [a31, a32, a33]])

def transform_coordinates(coords, params):
    """Apply transformation based on parameters"""
    mode = params['mode']
    transform_type = params['type']
    
    if mode == '2D':
        # Extract center point coordinates
        cx = params['cx']
        cy = params['cy']
        
        # Center the coordinates
        centered_coords = coords[:, :2] - np.array([cx, cy])
        
        if transform_type == 'helmert':
            # Create and apply transformation
            R = create_2d_helmert_matrix(params['rotation'])
            transformed = (1 + params['scale']) * (R @ centered_coords.T).T
            
        else:  # affine
            # Create and apply transformation
            A = create_2d_affine_matrix(
                params['a11'], params['a12'],
                params['a21'], params['a22']
            )
            transformed = (A @ centered_coords.T).T
            
        # Move back and apply translation
        xy_transformed = transformed + np.array([cx, cy]) + np.array([params['tx'], params['ty']])
        return np.column_stack((xy_transformed, coords[:, 2]))
        
    else:  # 3D mode
        # Extract center point coordinates
        cx = params['cx']
        cy = params['cy']
        cz = params['cz']
        
        # Center the coordinates
        centered_coords = coords - np.array([cx, cy, cz])
        
        if transform_type == 'helmert':
            # Create and apply transformation
            R = create_3d_helmert_matrix(params['omega'], params['phi'], params['kappa'])
            transformed = (1 + params['scale']) * (R @ centered_coords.T).T
            
        else:  # affine
            # Create and apply transformation
            A = create_3d_affine_matrix(
                params['a11'], params['a12'], params['a13'],
                params['a21'], params['a22'], params['a23'],
                params['a31'], params['a32'], params['a33']
            )
            transformed = (A @ centered_coords.T).T
            
        # Move back and apply translation
        return transformed + np.array([cx, cy, cz]) + np.array([params['tx'], params['ty'], params['tz']])


def transform_pointcloud(input_file, output_file, params):
    """Transform LAZ/LAS point cloud coordinates using PDAL"""
    try:
        # Define PDAL pipeline for reading the input file
        read_pipeline = [
            {
                "type": "readers.las",
                "filename": input_file
            }
        ]

        # Create and execute the read pipeline
        print("Reading input point cloud...")
        pipeline = pdal.Pipeline(json.dumps(read_pipeline))
        pipeline.execute()
        arrays = pipeline.arrays
        metadata = pipeline.metadata
        log = pipeline.log

        # Get point coordinates as numpy array
        print("Getting point coordinates...")
        points = arrays[0]
        coords = np.vstack((points['X'], points['Y'], points['Z'])).transpose()

        # Apply transformation
        print("Applying transformation...")
        transformed_coords = transform_coordinates(coords, params)

        # Update the point coordinates
        points['X'] = transformed_coords[:, 0]
        points['Y'] = transformed_coords[:, 1]
        points['Z'] = transformed_coords[:, 2]

        # Define PDAL pipeline for writing the output file
        write_pipeline = [
            {
                "type": "writers.las",
                "filename": output_file,
                "offset_x": "auto",
                "offset_y": "auto",
                "offset_z": "auto",
                "scale_x": 0.01,
                "scale_y": 0.01,
                "scale_z": 0.01
            }
        ]

        # Create and execute the write pipeline
        print("Writing output point cloud...")
        pipeline = pdal.Pipeline(json.dumps(write_pipeline), arrays=[points])
        pipeline.execute()

        print(f"Successfully transformed point cloud with {len(coords)} points")

    except Exception as e:
        print(f"Error transforming point cloud: {str(e)}")
        raise  # This will show the full error traceback
        sys.exit(1)

def transform_csv_data(input_file, output_file, params):
    """Transform CSV coordinate data"""
    try:
        print("Reading CSV data...")
        coords, ids, types = read_csv_coordinates(input_file)
        
        print("Applying transformation...")
        transformed_coords = transform_coordinates(coords, params)
        
        print("Saving transformed coordinates...")
        save_csv_coordinates(transformed_coords, ids, types, output_file)
        
        print(f"Successfully transformed {len(coords)} coordinate points")
        
    except Exception as e:
        print(f"Error transforming CSV data: {str(e)}")
        sys.exit(1)

def cleanup_dxf(doc):
    """Remove unnecessary sections and entries for better compatibility"""
    # Remove all classes for simplicity
    #if doc.classes is not None:
    #    doc.classes.clear()  
        
    # Simplify the OBJECTS section by removing complex dictionaries
    try:
        if 'ACAD_COLOR' in doc.rootdict:
            del doc.rootdict['ACAD_COLOR']
        if 'EZDXF_META' in doc.rootdict:
            del doc.rootdict['EZDXF_META']
        if 'ACAD_XREC_ROUNDTRIP' in doc.rootdict:
            del doc.rootdict['ACAD_XREC_ROUNDTRIP']
    except Exception as e:
        print(f"Warning: Could not remove some dictionary entries: {str(e)}")

    # Force older version compatibility
    doc.header['$ACADVER'] = 'AC1014'  # R14 version

def transform_point(point, transform_func):
    """
    Transform a single 3D point using the provided transformation function
    
    Args:
        point: Point coordinates as Vec3 or tuple (x,y,z)
        transform_func: Function that takes nx3 numpy array and returns transformed coordinates
    
    Returns:
        Vec3: Transformed point coordinates
    """
    try:
        # Convert point to numpy array format
        if isinstance(point, Vec3):
            coords = np.array([[point.x, point.y, point.z]])
        else:
            coords = np.array([[point[0], point[1], point[2] if len(point) > 2 else 0]])
            
        # Apply transformation
        transformed = transform_func(coords)[0]  # Take first point
        return Vec3(transformed)
        
    except Exception as e:
        print(f"Error transforming point {point}: {str(e)}")
        return point  # Return original point if transformation fails

def transform_polyline_points(points, transform_func):
    """
    Transform a list of polyline points
    
    Args:
        points: List of point coordinates (x,y) or (x,y,z)
        transform_func: Transformation function
    
    Returns:
        list: Transformed points with same structure as input
    """
    try:
        # Convert points to nx3 numpy array
        coords = []
        for p in points:
            if len(p) > 2:
                coords.append([p[0], p[1], p[2]])
            else:
                coords.append([p[0], p[1], 0])
                
        coords = np.array(coords)
        
        # Apply transformation
        transformed = transform_func(coords)
        
        # Convert back to original format
        result = []
        for i, p in enumerate(points):
            if len(p) > 2:
                result.append((transformed[i][0], transformed[i][1], transformed[i][2], *p[3:]))
            else:
                result.append((transformed[i][0], transformed[i][1], *p[2:]))
                
        return result
        
    except Exception as e:
        print(f"Error transforming polyline points: {str(e)}")
        return points

def transform_dxf_entity(entity, transform_func):
    """
    Transform a single DXF entity
    
    Args:
        entity: DXF entity object
        transform_func: Coordinate transformation function
    
    Returns:
        bool: True if transformation was successful
    """
    try:
        dxftype = entity.dxftype()
        
        if dxftype == 'POINT':
            # Transform point location
            new_location = transform_point(entity.dxf.location, transform_func)
            entity.dxf.location = new_location
            
        elif dxftype == 'LINE':
            # Transform start and end points
            entity.dxf.start = transform_point(entity.dxf.start, transform_func)
            entity.dxf.end = transform_point(entity.dxf.end, transform_func)
            
        elif dxftype == 'CIRCLE':
            # Transform center point
            entity.dxf.center = transform_point(entity.dxf.center, transform_func)
            
        elif dxftype == 'ARC':
            # Transform center point (radius and angles remain unchanged)
            entity.dxf.center = transform_point(entity.dxf.center, transform_func)
            
        elif dxftype == 'ELLIPSE':
            # Transform center and endpoints
            entity.dxf.center = transform_point(entity.dxf.center, transform_func)
            entity.dxf.major_axis = transform_point(
                Vec3(entity.dxf.major_axis), 
                transform_func
            ) - transform_point(Vec3((0,0,0)), transform_func)
            
        elif dxftype == 'LWPOLYLINE':
            # Transform lightweight polyline vertices
            with entity.points() as points:
                new_points = transform_polyline_points(points, transform_func)
                points[:] = new_points
                
        elif dxftype == 'POLYLINE':
            # Transform classic polyline vertices
            for vertex in entity.vertices:
                vertex.dxf.location = transform_point(vertex.dxf.location, transform_func)
                
        elif dxftype == 'SPLINE':
            # Transform control points and fit points
            new_control_points = [
                transform_point(p, transform_func) 
                for p in entity.control_points
            ]
            entity.control_points = new_control_points
            
            if entity.fit_points:
                new_fit_points = [
                    transform_point(p, transform_func) 
                    for p in entity.fit_points
                ]
                entity.fit_points = new_fit_points
                
        elif dxftype == '3DFACE':
            # Transform each vertex
            for i in range(4):
                point_attr = f'vtx{i}'
                if hasattr(entity.dxf, point_attr):
                    point = getattr(entity.dxf, point_attr)
                    new_point = transform_point(point, transform_func)
                    setattr(entity.dxf, point_attr, new_point)
                    
        elif dxftype == 'TEXT':
            # Transform insertion point and alignment point if present
            entity.dxf.insert = transform_point(entity.dxf.insert, transform_func)
            if hasattr(entity.dxf, 'align_point'):
                entity.dxf.align_point = transform_point(
                    entity.dxf.align_point, 
                    transform_func
                )
                
        elif dxftype == 'MTEXT':
            # Transform insertion point
            entity.dxf.insert = transform_point(entity.dxf.insert, transform_func)
            
        elif dxftype == 'INSERT':
            # Transform block reference insertion point
            entity.dxf.insert = transform_point(entity.dxf.insert, transform_func)
            
        elif dxftype == 'HATCH':
            # Transform hatch boundary paths
            for path in entity.paths:
                for vertex in path.vertices:
                    new_point = transform_point((vertex[0], vertex[1], 0), transform_func)
                    vertex = (new_point.x, new_point.y)
                    
        return True
        
    except Exception as e:
        print(f"Error transforming {dxftype} entity: {str(e)}")
        return False

def transform_dxf(input_file, output_file, params):
    """
    Transform DXF file coordinates
    
    Args:
        input_file: Path to input DXF file
        output_file: Path to output DXF file
        params: Dictionary of transformation parameters
    """
    try:
        print("Reading DXF file...")
        doc = ezdxf.readfile(input_file)
        
        # Create transformation function
        def transform_func(coords):
            return transform_coordinates(coords, params)
        
        # Transform entities in model space
        print("Transforming model space entities...")
        msp = doc.modelspace()
        entity_count = 0
        success_count = 0
        
        for entity in msp:
            entity_count += 1
            if transform_dxf_entity(entity, transform_func):
                success_count += 1
                
        # Transform entities in paper space
        print("Transforming paper space entities...")
        for layout in doc.layouts:
            if layout.name == 'Model': continue
            for entity in layout:
                entity_count += 1
                if transform_dxf_entity(entity, transform_func):
                    success_count += 1
                    
        # Transform block definitions
        print("Transforming blocks...")
        for block in doc.blocks:
            # Skip model and paper space blocks
            if block.name.lower() in ['*model_space', '*paper_space', '*paper_space0']:
                continue
                
            for entity in block:
                entity_count += 1
                if transform_dxf_entity(entity, transform_func):
                    success_count += 1
                    
        # Save transformed file
        print("Saving transformed DXF...")
        doc.saveas(output_file)
        
        print(f"Successfully transformed {success_count} out of {entity_count} entities")
        
    except Exception as e:
        print(f"Error transforming DXF file: {str(e)}")
        raise

def compute_corner_coordinates(transform, width, height):
    """Compute corner coordinates (pixel centers) of the raster"""
    return [
        transform * (0.5, 0.5),             # Upper-left (pixel center)
        transform * (width - 0.5, 0.5),     # Upper-right
        transform * (width - 0.5, height - 0.5),  # Lower-right
        transform * (0.5, height - 0.5)     # Lower-left
    ]


def compute_transformed_dimensions(transformed_corners, original_transform):
    """Calculate new raster dimensions based on transformed bounds and original resolution"""
    # Extract coordinates and find bounding box
    xs = [x for x, y in transformed_corners]
    ys = [y for x, y in transformed_corners]
    
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    
    # Get original pixel size from transform
    pixel_width = original_transform.a
    pixel_height = original_transform.e  # Negative value for north-up
    
    # Calculate new dimensions (add 1 pixel buffer)
    new_width = int(np.ceil((max_x - min_x) / abs(pixel_width))) + 1
    new_height = int(np.ceil((max_y - min_y) / abs(pixel_height))) + 1
    
    return (new_width, new_height), (min_x, max_y)

def transform_corners(corners, params):
    """Transform corner coordinates using given parameters"""
    # Convert corners to numpy array
    coords = np.array([[x, y, 0] for x, y in corners])
    
    # Apply transformation
    transformed = transform_coordinates(coords, params)
    
    # Return as list of (x, y) tuples
    return [(point[0], point[1]) for point in transformed]

# Define a Ray remote function for world coordinate conversion
@ray.remote
def parallel_coordinate_transform(output_coords_chunk, src_transform, new_transform, inverse_params):
    
    """Convert a chunk of output coordinates to world coordinates using Ray."""
    x = output_coords_chunk[:, 0]
    y = output_coords_chunk[:, 1]
    output_world_coords = np.column_stack([
        new_transform.a * x + new_transform.b * y + new_transform.c,
        new_transform.d * x + new_transform.e * y + new_transform.f
    ])
    output_world_3d = np.column_stack((output_world_coords, np.zeros(len(output_world_coords))))

    source_coords = transform_coordinates(output_world_3d, inverse_params)

    x = source_coords[:, 0]
    y = source_coords[:, 1]
    src_inv = ~src_transform
    source_pixels = np.column_stack([
        src_inv.a * x + src_inv.b * y + src_inv.c,
        src_inv.d * x + src_inv.e * y + src_inv.f
    ])
    
    return source_pixels
                            
def transform_geotiff(input_file, output_file, params):
    """
    Memory-efficient GeoTIFF transformation with timing analysis for each major section.
    Prints detailed timing information to help identify bottlenecks and parallelization opportunities.
    
    Args:
        input_file (str): Path to input GeoTIFF file
        output_file (str): Path to output GeoTIFF file
        params (dict): Dictionary containing transformation parameters
    """

    # Initialize Ray (call this once at the beginning of your script)
    ray.init()    

    # Block size for processing (adjust based on available memory)
    BLOCK_SIZE = 16384
    
    # Dictionary to store timing information
    timing_stats = {
        'initialization': 0,
        'dimension_calculation': 0,
        'block_processing': {},  # Will store per-band timing
        'block_processing_details': {},  # Will store detailed timing for first block of each band
        'total_time': 0
    }
    
    total_start_time = time.time()
    
    try:
        # SECTION 1: Initialization and File Opening
        section_start = time.time()
        print("\n=== Starting GeoTIFF Transformation ===")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with rasterio.open(input_file) as src:
            src_meta = src.meta.copy()
            src_transform = src.transform
            
            if src.nodata is None:
                nodata = 0 if src.count in [3, 4] else -32767
            else:
                nodata = src.nodata
                
        initialization_time = time.time() - section_start
        timing_stats['initialization'] = initialization_time
        print(f"\n[Timing] Initialization: {initialization_time:.2f} seconds")
        print(f"- File opened and metadata read")
        
        # SECTION 2: Calculate Output Dimensions
        section_start = time.time()
        
        with rasterio.open(input_file) as src:
            corners = np.array([
                [0, 0],
                [src.width-1, 0],
                [src.width-1, src.height-1],
                [0, src.height-1],
            ])
            
            world_corners = np.array([src.xy(row, col) for row, col in corners])
            world_corners_3d = np.column_stack((world_corners, np.zeros(len(world_corners))))
            transformed_corners = transform_coordinates(world_corners_3d, params)
            transformed_corners_2d = transformed_corners[:, :2]
            
            # Calculate new bounds and dimensions
            min_x = transformed_corners_2d[:, 0].min()
            max_x = transformed_corners_2d[:, 0].max()
            min_y = transformed_corners_2d[:, 1].min()
            max_y = transformed_corners_2d[:, 1].max()
            
            res_x = abs(src_transform[0])
            res_y = abs(src_transform[4])
            
            new_width = int(np.ceil((max_y - min_y) / res_y))
            new_height = int(np.ceil((max_x - min_x) / res_x))
            
            # Create new geotransform
            new_transform = from_origin(min_x, max_y, res_x, res_y)
            
            # Update metadata for output
            dst_meta = src_meta.copy()
            dst_meta.update({
                'height': new_height,
                'width': new_width,
                'transform': new_transform,
                'nodata': nodata,
                'BIGTIFF': 'YES',
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'predictor': 2,
                'driver': 'GTiff'
            })
        
        dimension_calc_time = time.time() - section_start
        timing_stats['dimension_calculation'] = dimension_calc_time
        print(f"\n[Timing] Dimension Calculation: {dimension_calc_time:.2f} seconds")
        print(f"- Input dimensions: {src.width}x{src.height}")
        print(f"- Output dimensions: {new_width}x{new_height}")
        
        # SECTION 3: Prepare Inverse Transform
        section_start = time.time()
        inverse_params = params.copy()
        if params['type'] == 'helmert':
            if params['mode'] == '2D':
                inverse_params.update({
                    'rotation': -params['rotation'],
                    'scale': -params['scale'],
                    'tx': -params['tx'],
                    'ty': -params['ty']
                })
        else:  # affine
            if params['mode'] == '2D':
                det = params['a11'] * params['a22'] - params['a12'] * params['a21']
                inverse_params.update({
                    'a11': params['a22'] / det,
                    'a12': -params['a12'] / det,
                    'a21': -params['a21'] / det,
                    'a22': params['a11'] / det,
                    'tx': -(params['tx'] * inverse_params['a11'] + params['ty'] * inverse_params['a12']),
                    'ty': -(params['tx'] * inverse_params['a21'] + params['ty'] * inverse_params['a22'])
                })
        
        inverse_transform_time = time.time() - section_start
        print(f"\n[Timing] Inverse Transform Preparation: {inverse_transform_time:.2f} seconds")
        
        # SECTION 4: Process Blocks
        print("\n=== Starting Block Processing ===")
        with rasterio.open(input_file) as src, rasterio.open(output_file, 'w', **dst_meta) as dst:
            n_blocks_x = int(np.ceil(new_width / BLOCK_SIZE))
            n_blocks_y = int(np.ceil(new_height / BLOCK_SIZE))
            total_blocks = n_blocks_x * n_blocks_y
            
            print(f"Total blocks to process: {total_blocks} ({n_blocks_x}x{n_blocks_y})")
            print(f"Block size: {BLOCK_SIZE}x{BLOCK_SIZE} pixels")
            
            for band_idx in range(1, src.count + 1):
                band_start_time = time.time()
                print(f"\nProcessing band {band_idx}/{src.count}")
                timing_stats['block_processing_details'][band_idx] = {}
                
                block_times = []  # Store processing time for each block
                
                # Read source band just outside of the block loop
                reading_start = time.time()
                source_data = src.read(band_idx)
                source_nodata_mask = source_data == nodata
                source_data_float = source_data.astype(np.float64)
                source_data_float[source_nodata_mask] = np.nan
                reading_time = time.time() - reading_start
                
                with tqdm(total=total_blocks, desc=f"Band {band_idx}") as pbar:
                    for block_y in range(n_blocks_y):
                        for block_x in range(n_blocks_x):
                            block_start_time = time.time()
                            
                            # Calculate block dimensions
                            block_width = min(BLOCK_SIZE, new_width - block_x * BLOCK_SIZE)
                            block_height = min(BLOCK_SIZE, new_height - block_y * BLOCK_SIZE)
                            
                            # SUBSECTION 4.1: Create coordinate grid
                            grid_start = time.time()
                            block_rows, block_cols = np.mgrid[
                                block_y * BLOCK_SIZE:block_y * BLOCK_SIZE + block_height,
                                block_x * BLOCK_SIZE:block_x * BLOCK_SIZE + block_width
                            ]
                            output_coords = np.stack((block_cols.flatten(), block_rows.flatten()), axis=1)
                            grid_time = time.time() - grid_start
                            
                            # SUBSECTION 4.2: Coordinate transformation
                            transform_start = time.time()

                            # Split the output_coords into chunks for parallel processing
                            num_chunks = ray.available_resources().get('CPU', 1)  # Use available CPUs
                            chunks = np.array_split(output_coords, num_chunks)                            

                            # Use Ray to parallelize the coordinate transformation
                            futures = [parallel_coordinate_transform.remote(chunk, src_transform, new_transform, inverse_params) for chunk in chunks]
                            source_pixel_chunks = ray.get(futures)
                            
                            # Combine the results
                            source_pixels = np.vstack(source_pixel_chunks)                            
                            
                            transform_time = time.time() - transform_start
                            
                            # SUBSECTION 4.3: Interpolation
                            interp_start = time.time()
                            
                            valid_x = (source_pixels[:, 0] >= 0) & (source_pixels[:, 0] < src.width)
                            valid_y = (source_pixels[:, 1] >= 0) & (source_pixels[:, 1] < src.height)
                            valid_pixels = valid_x & valid_y
                            
                            block_data = np.full((block_height, block_width), nodata, dtype=src.dtypes[0])
                            
                            if np.any(valid_pixels):
                                #print("---read source band")
                                #source_data = src.read(band_idx)
                                #source_nodata_mask = source_data == nodata
                                #source_data_float = source_data.astype(np.float64)
                                #source_data_float[source_nodata_mask] = np.nan
                                
                                valid_source_pixels = source_pixels[valid_pixels]
                                
                                resampled_values = map_coordinates(
                                    source_data_float,
                                    [valid_source_pixels[:, 1], valid_source_pixels[:, 0]],
                                    order=1,
                                    mode='constant',
                                    cval=np.nan
                                )
                                
                                resampled_values = np.nan_to_num(resampled_values, nan=nodata)
                                block_data_flat = block_data.ravel()
                                block_data_flat[valid_pixels] = resampled_values
                                block_data = block_data_flat.reshape((block_height, block_width))
                            
                            interp_time = time.time() - interp_start
                            
                            # SUBSECTION 4.4: Data writing
                            writing_start = time.time()
                            dst.write(
                                block_data,
                                band_idx,
                                window=Window(
                                    block_x * BLOCK_SIZE,
                                    block_y * BLOCK_SIZE,
                                    block_width,
                                    block_height
                                )
                            )
                            
                            writing_time = time.time() - writing_start
                            
                            # Store timing for first block of each band
                            if block_x == 0 and block_y == 0:
                                timing_stats['block_processing_details'][band_idx] = {
                                    'grid_creation': grid_time,
                                    #'world_coords': world_time,
                                    'transformation': transform_time,
                                    #'pixel_conversion': pixel_time,
                                    'interpolation': interp_time,
                                    'writing': writing_time,
                                    'total_block_time': time.time() - block_start_time
                                }
                            
                            block_times.append(time.time() - block_start_time)
                            pbar.update(1)
                
                # Store band processing statistics
                band_time = time.time() - band_start_time
                timing_stats['block_processing'][band_idx] = {
                    'total_time': band_time,
                    'average_block_time': np.mean(block_times),
                    'min_block_time': np.min(block_times),
                    'max_block_time': np.max(block_times)
                }
                
                # Copy band description
                dst.set_band_description(band_idx, src.descriptions[band_idx-1] or '')
        
        # Calculate and print final statistics
        total_time = time.time() - total_start_time
        timing_stats['total_time'] = total_time
        
        print("\n=== Timing Analysis ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Initialization: {timing_stats['initialization']:.2f} seconds ({(timing_stats['initialization']/total_time)*100:.1f}%)")
        print(f"Dimension calculation: {timing_stats['dimension_calculation']:.2f} seconds ({(timing_stats['dimension_calculation']/total_time)*100:.1f}%)")
        
        print("\nPer-band processing times:")
        for band_idx, stats in timing_stats['block_processing'].items():
            print(f"\nBand {band_idx}:")
            print(f"- Total time: {stats['total_time']:.2f} seconds ({(stats['total_time']/total_time)*100:.1f}%)")
            print(f"- Data reading time: {stats['average_block_time']:.3f} seconds")
            print(f"- Average block time: {reading_time:.3f} seconds")
            print(f"- Min block time: {stats['min_block_time']:.3f} seconds")
            print(f"- Max block time: {stats['max_block_time']:.3f} seconds")
            
            # Print detailed timing for first block
            block_details = timing_stats['block_processing_details'][band_idx]
            print(f"\nDetailed timing for first block of Band {band_idx}:")
            print(f"- Grid creation: {block_details['grid_creation']:.3f} seconds")
            print(f"- Transformation: {block_details['transformation']:.3f} seconds")
            print(f"- Interpolation: {block_details['interpolation']:.3f} seconds")
            print(f"- Writing: {block_details['writing']:.3f} seconds")
        
        print(f"\nSuccessfully transformed GeoTIFF: {new_width}x{new_height} pixels")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error transforming GeoTIFF: {str(e)}")
        raise

    finally:
        ray.shutdown()

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("\nUsage: python transform_coords.py input_file output_file transform_params.txt")
        print("Supported formats: .csv, .laz, .las, .dxf, .tif, .tiff")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    param_file = sys.argv[3]
    
    # Get file extensions
    input_ext = os.path.splitext(input_file)[1].lower()
    output_ext = os.path.splitext(output_file)[1].lower()
    
    # Validate file extensions
    valid_extensions = {'.csv', '.laz', '.las', '.dxf', '.tif', '.tiff'}
    if input_ext not in valid_extensions:
        print(f"Error: Input file must be one of: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    if output_ext not in valid_extensions:
        print(f"Error: Output file must be one of: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    if input_ext != output_ext:
        print("Error: Input and output files must be of the same type")
        sys.exit(1)
    
    # Group tif/tiff as same type for comparison
    def get_file_type(ext):
        if ext in ['.tif', '.tiff']:
            return 'geotiff'
        return ext[1:]  # Remove the dot
    
    if get_file_type(input_ext) != get_file_type(output_ext):
        print("Error: Input and output files must be of the same type")
        sys.exit(1)
    
    # Validate input files exist
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    if not os.path.exists(param_file):
        print(f"Error: Parameter file '{param_file}' not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Read transformation parameters
        params = read_params(param_file)
        
        # Process based on file type
        if input_ext == '.csv':
            print("\nProcessing CSV file...")
            transform_csv_data(input_file, output_file, params)
        elif input_ext == '.dxf':
            print("\nProcessing DXF file...")
            transform_dxf(input_file, output_file, params)
        elif input_ext in ['.tif', '.tiff']:
            print("\nProcessing GeoTIFF file...")
            transform_geotiff(input_file, output_file, params)
        else:  # .laz or .las
            print("\nProcessing point cloud file...")
            transform_pointcloud(input_file, output_file, params)
        
        print(f"\nTransformation completed successfully!")
        print(f"Transformed data saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError during transformation: {str(e)}")
        sys.exit(1)

    
if __name__ == "__main__":
    main()