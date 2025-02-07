# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:28:42 2025

@author: dnugr
"""

# -*- coding: utf-8 -*-
"""
Coordinate transformation script supporting CSV, LAZ/LAS, DXF, and GeoTIFF files
Usage: python transform_coords.py input_file output_file transform_params.txt
Input can be .csv, .laz/.las, .dxf, or .tif/.tiff
"""

import numpy as np
import pdal
import json
import csv
import sys
import os
from pathlib import Path
import laspy
import ezdxf
from ezdxf.math import Vec3
from ezdxf.math import Matrix44
from ezdxf import transform
import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.warp import reproject, Resampling  # Add these imports at the top
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

def create_transformation_matrix(file_path):
    
    """Read transformation parameters from file"""       
    
    # Read the parameters from the file
    with open(file_path, 'r') as file:
        params = {}
        for line in file:
            print(line)

            key, value = line.strip().split('=')

            # Convert numeric values to float
            if key not in ['mode', 'type']:
                value = float(value)
            params[key] = value
            print(key,value)

    # Determine the mode and type
    mode = params.get('mode')
    transform_type = params.get('type')

    if mode == '3D':
        if transform_type == 'helmert':
            # Extract Helmert parameters
            tx, ty, tz = params['tx'], params['ty'], params['tz']
            scale = 1 + params['scale']
            omega, phi, kappa = params['omega'], params['phi'], params['kappa']

            # Create rotation matrix from omega, phi, kappa (in radians)
            # Assuming small angles, so we can approximate the rotation matrix
            # as a combination of small angle rotations
            Rx = Matrix44.x_rotate(omega)
            Ry = Matrix44.y_rotate(phi)
            Rz = Matrix44.z_rotate(kappa)
            rotation_matrix = Rx @ Ry @ Rz

            # Create scaling matrix
            scaling_matrix = Matrix44.scale(scale, scale, scale)

            # Create translation matrix
            translation_matrix = Matrix44.translate(tx, ty, tz)

            # Combine all transformations: T * R * S
            transformation_matrix = translation_matrix @ rotation_matrix @ scaling_matrix

        elif transform_type == 'affine':
            # Extract Affine parameters
            tx, ty, tz = params['tx'], params['ty'], params['tz']
            a11, a12, a13 = params['a11'], params['a12'], params['a13']
            a21, a22, a23 = params['a21'], params['a22'], params['a23']
            a31, a32, a33 = params['a31'], params['a32'], params['a33']

            # Create the affine transformation matrix
            transformation_matrix = Matrix44([
                [a11, a12, a13, tx],
                [a21, a22, a23, ty],
                [a31, a32, a33, tz],
                [0, 0, 0, 1]
            ])

        else:
            raise ValueError(f"Unsupported transformation type: {transform_type}")

    elif mode == '2D':
        if transform_type == 'helmert':
            # Extract Helmert parameters
            tx, ty = params['tx'], params['ty']
            scale = 1 + params['scale']
            rotation = params['rotation']

            # Create 2D rotation matrix
            rotation_matrix = Matrix44.z_rotate(rotation)

            # Create 2D scaling matrix
            scaling_matrix = Matrix44.scale(scale, scale, 1)

            # Create 2D translation matrix
            translation_matrix = Matrix44.translate(tx, ty, 0)

            # Combine all transformations: T * R * S
            transformation_matrix = translation_matrix @ rotation_matrix @ scaling_matrix

        elif transform_type == 'affine':
            # Extract Affine parameters
            tx, ty = params['tx'], params['ty']
            a11, a12 = params['a11'], params['a12']
            a21, a22 = params['a21'], params['a22']

            # Create the 2D affine transformation matrix
            transformation_matrix = Matrix44([
                [a11, a12, 0, tx],
                [a21, a22, 0, ty],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        else:
            raise ValueError(f"Unsupported transformation type: {transform_type}")

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return transformation_matrix


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

def transform_dxf_entity(entity, transform_func):
    """Transform a single DXF entity's coordinates"""
    if entity.dxftype() == 'POINT':
        # Transform point coordinates
        
        location = entity.dxf.location
        x, y, z = location
        
        coords = np.array([[x, y, z]])
        new_coords = transform_func(coords)[0]

        x = new_coords[0]
        y = new_coords[1]
        z = new_coords[2]
        
        location = x, y, z
        entity.dxf.location = location
        
        
    elif entity.dxftype() == 'LINE':
        # Transform start and end points
        start_coords = np.array([[entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z]])
        end_coords = np.array([[entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z]])
        
        new_start = transform_func(start_coords)[0]
        new_end = transform_func(end_coords)[0]
        
        entity.dxf.start = Vec3(new_start)
        entity.dxf.end = Vec3(new_end)
        
    elif entity.dxftype() == 'LWPOLYLINE':
        # Transform polyline vertices
        with entity.points() as points:
            coords = np.array([[p[0], p[1], 0] for p in points])
            new_coords = transform_func(coords)
            # Update points (keeping any bulge values)
            for i, p in enumerate(points):
                points[i] = (new_coords[i][0], new_coords[i][1], *p[2:])
                
    elif entity.dxftype() == 'POLYLINE':
        # Transform 3D polyline vertices
        for vertex in entity.vertices:
            coords = np.array([[vertex.dxf.location.x, vertex.dxf.location.y, vertex.dxf.location.z]])
            new_coords = transform_func(coords)[0]
            vertex.dxf.location = Vec3(new_coords)
            
    elif entity.dxftype() == '3DFACE':
        # Transform each corner of the 3D face
        for i in range(4):
            point_attr = f'vtx{i}'
            if hasattr(entity.dxf, point_attr):
                point = getattr(entity.dxf, point_attr)
                coords = np.array([[point.x, point.y, point.z]])
                new_coords = transform_func(coords)[0]
                setattr(entity.dxf, point_attr, Vec3(new_coords))
                
    elif entity.dxftype() == 'INSERT':
        # Transform block reference insertion point
        coords = np.array([[entity.dxf.insert.x, entity.dxf.insert.y, entity.dxf.insert.z]])
        new_coords = transform_func(coords)[0]
        entity.dxf.insert = Vec3(new_coords)
        
    elif entity.dxftype() == 'TEXT':
        # Transform text insertion point
        coords = np.array([[entity.dxf.insert.x, entity.dxf.insert.y, 
                           getattr(entity.dxf, 'z', 0)]])  # Z might not exist
        new_coords = transform_func(coords)[0]
        entity.dxf.insert = Vec3(new_coords)
        
    elif entity.dxftype() == 'CIRCLE':
        # Transform circle center
        coords = np.array([[entity.dxf.center.x, entity.dxf.center.y, entity.dxf.center.z]])
        new_coords = transform_func(coords)[0]
        entity.dxf.center = Vec3(new_coords)
        
    elif entity.dxftype() == 'ARC':
        # Transform arc center
        coords = np.array([[entity.dxf.center.x, entity.dxf.center.y, entity.dxf.center.z]])
        new_coords = transform_func(coords)[0]
        entity.dxf.center = Vec3(new_coords)

def transform_dxf_old(input_file, output_file, params):
    """Transform DXF file coordinates"""
    try:
        print("Reading DXF file...")
        doc = ezdxf.readfile(input_file)
        
        # Create a transformation function that applies our parameters
        def transform_func(coords):
            return transform_coordinates(coords, params)
        
        print("Transforming entities...")
        # Process all model space entities
        msp = doc.modelspace()
        entity_count = 0
        
        # Transform each entity
        for entity in msp:
            try:
                transform_dxf_entity(entity, transform_func)
                entity_count += 1
            except Exception as e:
                print(f"Warning: Could not transform entity {entity.dxftype()}: {str(e)}")
                continue
        
        # Process all blocks
        print("Transforming blocks...")
        for block in doc.blocks:
            # Skip model space and paper space blocks
            if block.name.lower() in ['*model_space', '*paper_space']:
                continue
                
            for entity in block:
                try:
                    transform_dxf_entity(entity, transform_func)
                    entity_count += 1
                except Exception as e:
                    print(f"Warning: Could not transform block entity {entity.dxftype()}: {str(e)}")
                    continue
        
        print("Saving transformed DXF...")
        doc.saveas(output_file, fmt="asc")
        
        print(f"Successfully transformed {entity_count} entities")
            
    except Exception as e:
        print(f"Error transforming DXF file: {str(e)}")
        raise  # This will show the full error traceback
        sys.exit(1)

def transform_dxf(input_file, output_file, param_file):
    # Load the DXF file
    doc = ezdxf.readfile(input_file)
    msp = doc.modelspace()
    
    try:
        transform_matrix = create_transformation_matrix(param_file)
    
    except Exception as e:
        print(f"Error transforming DXF file: {str(e)}")
        raise  # This will show the full error traceback
        sys.exit(1)
    
    # Iterate through entities and apply transformations
    for entity in msp:
        print(entity)
        #entity.transform(transform_matrix)
        #apply_affine_transform_to_entity2(entity, transform_matrix)
    
    log = transform.inplace(msp, transform_matrix)
    
    print(log.messages())

    # Save the modified DXF file
    doc.saveas(output_file, fmt="asc")

def compute_corner_coordinates(transform, width, height):
    """Compute corner coordinates (pixel centers) of the raster"""
    return [
        transform * (0.5, 0.5),             # Upper-left (pixel center)
        transform * (width - 0.5, 0.5),     # Upper-right
        transform * (width - 0.5, height - 0.5),  # Lower-right
        transform * (0.5, height - 0.5)     # Lower-left
    ]

def transform_geotiff(input_file, output_file, params):
    """Transform GeoTIFF coordinates with correct geotransform and dimensions"""
    try:
        with rasterio.open(input_file) as src:
            data = src.read()
            original_transform = src.transform
            crs = src.crs
            dtype = src.dtypes[0]
            count = src.count

            # Get original pixel size
            pixel_width = original_transform.a
            pixel_height = original_transform.e  # Negative for north-up

            # Calculate original corners (pixel centers)
            corners = compute_corner_coordinates(src.transform, src.width, src.height)
            
            # Transform corners using the coordinate transformation
            transformed_corners = transform_corners(corners, params)
            xs = [x for x, y in transformed_corners]
            ys = [y for x, y in transformed_corners]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Calculate new transform origin (top-left of upper-left pixel)
            new_west = min_x - (pixel_width / 2)
            new_north = max_y - (pixel_height / 2)  # pixel_height is negative
            new_transform = from_origin(new_west, new_north, pixel_width, abs(pixel_height))

            # Calculate new dimensions to cover all transformed corners
            new_width = int(np.ceil((max_x - min_x + pixel_width) / pixel_width))
            new_height = int(np.ceil((max_y - min_y + abs(pixel_height)) / abs(pixel_height)))

            # Update output profile
            out_profile = src.profile.copy()
            out_profile.update({
                'transform': new_transform,
                'width': new_width,
                'height': new_height
            })

            with rasterio.open(output_file, 'w', **out_profile) as dst:
                reproject(
                    source=data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear
                )
                print(count)
                if count > 1:
                    for band in range(1,count+1):
                        print(band)
                        dst.write(src.read(band), band)
                
                dst.update_tags(**src.tags())
                dst.descriptions = src.descriptions

        print(f"Transformed GeoTIFF created: {new_width}x{new_height} pixels")

    except Exception as e:
        print(f"Error transforming GeoTIFF: {str(e)}")
        sys.exit(1)

def compute_geotransform(transformed_corners, original_transform):
    """Compute new geotransform from transformed coordinates"""
    xs = [x for x, y in transformed_corners]
    ys = [y for x, y in transformed_corners]
    
    return Affine(
        original_transform.a,  # Pixel width (x-scale)
        original_transform.b,  # X-shear
        min(xs),               # Upper-left X
        original_transform.d,  # Y-shear
        original_transform.e,  # Pixel height (y-scale, typically negative)
        max(ys)                # Upper-left Y
    )


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

def transform_geotiff(input_file, output_file, params):
    """Transform GeoTIFF coordinates using affine parameters from file"""
    try:
        # Extract affine parameters from params
        a = params.get('a11', 1.0)
        b = params.get('a12', 0.0)
        c = params.get('tx', 0.0)
        d = params.get('a21', 0.0)
        e = params.get('a22', 1.0)
        f = params.get('ty', 0.0)

        def affine_transform(x, y):
            """Apply affine transformation using parameters from file"""
            return (
                a * x + b * y + c,
                d * x + e * y + f
            )

        with rasterio.open(input_file) as src:
            # Get original properties
            data = src.read()
            profile = src.profile
            orig_transform = src.transform
            height, width = src.shape

            # Transform corner coordinates
            corners = [
                (src.bounds.left, src.bounds.top),
                (src.bounds.right, src.bounds.top),
                (src.bounds.left, src.bounds.bottom),
                (src.bounds.right, src.bounds.bottom)
            ]

            transformed_corners = [affine_transform(x, y) for x, y in corners]

            # Calculate new bounds
            xs = [x for x, y in transformed_corners]
            ys = [y for x, y in transformed_corners]
            new_left = min(xs)
            new_right = max(xs)
            new_top = max(ys)
            new_bottom = min(ys)

            # Create new transform matrix
            new_transform = Affine(
                (new_right - new_left)/width,  # a: pixel width
                orig_transform.b,             # b: rotation (keep original)
                new_left,                     # c: x origin
                orig_transform.d,             # d: rotation (keep original)
                (new_bottom - new_top)/height,# e: pixel height (negative)
                new_top                       # f: y origin
            )

            # Update profile with new transform
            profile.update({
                'transform': new_transform,
                'compress': 'lzw'
            })

            # Create destination array
            dest = np.zeros_like(data)

            # Calculate new dimensions
            new_width = int(np.ceil((new_right - new_left) / abs(new_transform.a)))
            new_height = int(np.ceil((new_top - new_bottom) / abs(new_transform.e)))

            # Write output using array warping
            with rasterio.open(output_file, 'w', 
                             width=new_width,
                             height=new_height,
                             **profile) as dst:
                reproject(
                    source=data,
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )
                dst.write(dest)

        print(f"Transformed GeoTIFF created with parameters: a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")

    except Exception as e:
        print(f"Error transforming GeoTIFF: {str(e)}")
        sys.exit(1)

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