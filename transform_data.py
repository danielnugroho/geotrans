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
import ezdxf
from ezdxf.math import Vec3
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