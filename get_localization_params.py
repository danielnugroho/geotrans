# -*- coding: utf-8 -*-

__version__ = "1.1.0"
__author__ = "Daniel Adi Nugroho"
__email__ = "dnugroho@gmail.com"
__status__ = "Production"
__date__ = "2025-02-12"
__copyright__ = "Copyright (c) 2025 Daniel Adi Nugroho"
__license__ = "GNU General Public License v3.0 (GPL-3.0)"

# Version History
# --------------

# 1.1.0 (2025-02-12)
# - First fully functional version, with parallel processing implemented
# - Source code cleanup and modification to GNU-GPL license instead of MIT

# 1.0.0 (2025-02-07)
# - Initial release
# - Support for 2D and 3D transformations
# - Helmert and Affine transformation options
# - CSV input/output functionality
# - Error analysis and statistics


"""
Coordinate Transformation Computation Module
==============================

This module provides functionality for transforming coordinates between different coordinate systems
using either Helmert or Affine transformations in both 2D and 3D space.

Purpose:
--------
- Transform coordinates from one system to another (e.g., local to global coordinates)
- Support both 2D and 3D coordinate transformations
- Provide both Helmert (similarity) and Affine transformation options
- Calculate and report transformation accuracy statistics

Requirements:
------------
- Python 3.10 or higher
- Required packages: numpy, scipy
- Input data must be in CSV format

Input CSV Format:
---------------
The input CSV files (both source and target) must have the following columns:
- ID: Point identifier
- X: X coordinate
- Y: Y coordinate
- Z: Z coordinate
- Type: Point type (optional)

Usage:
-----
1. Prepare two CSV files:
   - Source coordinates CSV (points in original coordinate system)
   - Target coordinates CSV (same points in desired coordinate system)

2. Run the script:
   python get_localization_params.py

3. When prompted:
   - Enter the path to source coordinates CSV
   - Enter the path to target coordinates CSV
   - Choose transformation mode (2D or 3D)
   - Choose transformation type (Helmert or Affine)
   - Choose whether to save transformation parameters

Transformation Types:
------------------
1. Helmert (Similarity) Transformation:
   - 2D: 4 parameters (2 translations, 1 scale, 1 rotation)
   - 3D: 7 parameters (3 translations, 1 scale, 3 rotations)
   - Preserves angles and shape
   - Minimum points required: 2 for 2D, 3 for 3D

2. Affine Transformation:
   - 2D: 6 parameters (2 translations, 4 transformation coefficients)
   - 3D: 12 parameters (3 translations, 9 transformation coefficients)
   - Allows for different scales in different directions
   - Minimum points required: 3 for 2D, 4 for 3D

Output:
-------
- Transformation parameters
- Detailed error analysis including RMS errors
- Sample results showing original, transformed, and target coordinates
- Option to save parameters to a file

Example Usage:
------------
>>> python get_localization_params.py
Enter path to source coordinates CSV: source_points.csv
Enter path to target coordinates CSV: target_points.csv
Choose transformation mode (2D/3D): 3D
Choose transformation type (helmert/affine): helmert



GNU GENERAL PUBLIC LICENSE
--------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""

import numpy as np
from scipy.optimize import least_squares
import csv

def read_csv(file_path):
    """
    Read and validate CSV file
    Expects columns: ID, X, Y, Z, Type
    """
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            
            # Read header
            try:
                header = next(csv_reader)
            except StopIteration:
                raise ValueError("CSV file is empty")
            
            # Validate minimum required columns
            if len(header) < 4:
                raise ValueError("CSV must have at least 4 columns (ID, X, Y, Z)")
            
            data = []
            line_number = 2  # Start from line 2 (after header)
            
            for row in csv_reader:
                try:
                    if len(row) < 4:
                        raise ValueError(f"Row {line_number} has insufficient columns")
                    
                    # Convert coordinates to float
                    coords = [float(row[1]), float(row[2]), float(row[3])]
                    data.append(coords)
                    
                except ValueError as e:
                    if "could not convert string to float" in str(e):
                        raise ValueError(f"Invalid coordinate value in row {line_number}")
                    else:
                        raise
                        
                line_number += 1
            
            if not data:
                raise ValueError("No valid coordinate data found in CSV")
            
            return np.array(data)
            
    except csv.Error as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def create_2d_helmert_matrix(theta):
    """Create 2D Helmert rotation matrix"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

def create_2d_affine_matrix(params):
    """
    Create 2D affine transformation matrix
    params: [a11, a12, a21, a22] - transformation coefficients
    """
    return np.array([[params[0], params[1]],
                    [params[2], params[3]]])

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

def create_3d_affine_matrix(params):
    """
    Create 3D affine transformation matrix
    params: [a11, a12, a13, a21, a22, a23, a31, a32, a33] - transformation coefficients
    """
    return np.array([[params[0], params[1], params[2]],
                    [params[3], params[4], params[5]],
                    [params[6], params[7], params[8]]])

def transform_points(params, source, mode='3D', transform_type='helmert'):
    """
    Apply transformation based on mode and type
    
    Helmert parameters:
    2D: [tx, ty, s, theta, cx, cy]
    3D: [tx, ty, tz, s, omega, phi, kappa, cx, cy, cz]
    
    Affine parameters:
    2D: [tx, ty, a11, a12, a21, a22, cx, cy]
    3D: [tx, ty, tz, a11, a12, a13, a21, a22, a23, a31, a32, a33, cx, cy, cz]
    """
    if mode == '2D':
        if transform_type == 'helmert':
            tx, ty, s, theta, cx, cy = params
            
            # Center the source data
            centered_source = source[:, :2] - np.array([cx, cy])
            
            # Create and apply transformation
            R = create_2d_helmert_matrix(theta)
            transformed = (1 + s) * (R @ centered_source.T).T
            
        else:  # affine
            tx, ty, a11, a12, a21, a22, cx, cy = params
            
            # Center the source data
            centered_source = source[:, :2] - np.array([cx, cy])
            
            # Create and apply transformation
            A = create_2d_affine_matrix([a11, a12, a21, a22])
            transformed = (A @ centered_source.T).T
            
        # Move back and apply translation
        xy_transformed = transformed + np.array([cx, cy]) + np.array([tx, ty])
        return np.column_stack((xy_transformed, source[:, 2]))
        
    else:  # 3D mode
        if transform_type == 'helmert':
            tx, ty, tz, s, omega, phi, kappa, cx, cy, cz = params
            
            # Center the source data
            centered_source = source - np.array([cx, cy, cz])
            
            # Create and apply transformation
            R = create_3d_helmert_matrix(omega, phi, kappa)
            transformed = (1 + s) * (R @ centered_source.T).T
            
        else:  # affine
            tx, ty, tz, a11, a12, a13, a21, a22, a23, a31, a32, a33, cx, cy, cz = params
            
            # Center the source data
            centered_source = source - np.array([cx, cy, cz])
            
            # Create and apply transformation
            A = create_3d_affine_matrix([a11, a12, a13, a21, a22, a23, a31, a32, a33])
            transformed = (A @ centered_source.T).T
            
        # Move back and apply translation
        return transformed + np.array([cx, cy, cz]) + np.array([tx, ty, tz])

def residuals(params, source, target, mode='3D', transform_type='helmert'):
    """Calculate residuals based on mode and transformation type"""
    transformed = transform_points(params, source, mode, transform_type)
    if mode == '2D':
        return (transformed[:, :2] - target[:, :2]).flatten()
    else:
        return (transformed - target).flatten()
    
def check_minimum_points(n_points, mode, transform_type):
    """
    Check if number of point pairs is sufficient for the requested transformation.
    
    Args:
        n_points: number of point pairs available
        mode: '2D' or '3D'
        transform_type: 'helmert' or 'affine'
    
    Raises:
        ValueError: if insufficient point pairs are provided
    """
    min_points = {
        '2D': {
            'helmert': 2,  # 4 equations (2 per point) for 4 parameters
            'affine': 3    # 6 equations (2 per point) for 6 parameters
        },
        '3D': {
            'helmert': 3,  # 9 equations (3 per point) for 7 parameters
            'affine': 4    # 12 equations (3 per point) for 12 parameters
        }
    }
    
    required = min_points[mode][transform_type]
    if n_points < required:
        error_messages = {
            '2D': {
                'helmert': "2D Helmert transformation requires at least 2 point pairs (provides 4 equations for 4 parameters: 1 scale, 1 rotation, 2 translations)",
                'affine': "2D Affine transformation requires at least 3 point pairs (provides 6 equations for 6 parameters: 4 transformation coefficients, 2 translations)"
            },
            '3D': {
                'helmert': "3D Helmert transformation requires at least 3 point pairs (provides 9 equations for 7 parameters: 1 scale, 3 rotations, 3 translations)",
                'affine': "3D Affine transformation requires at least 4 point pairs (provides 12 equations for 12 parameters: 9 transformation coefficients, 3 translations)"
            }
        }
        raise ValueError(f"\n{error_messages[mode][transform_type]}\nProvided: {n_points} point pairs")


def compute_transformation(source, target, mode='3D', transform_type='helmert'):
    """
    Compute transformation parameters based on mode and type
    """
    
    # Check if we have enough point pairs
    n_points = len(source)
    check_minimum_points(n_points, mode, transform_type)
    
    # Use centroids as initial guess for center
    initial_cx = np.mean(source[:, 0])
    initial_cy = np.mean(source[:, 1])
    
    if mode == '2D':
        if transform_type == 'helmert':
            initial_params = [0, 0,    # translations
                            0,         # scale
                            0,         # rotation
                            initial_cx, initial_cy]  # center
        else:  # affine
            initial_params = [0, 0,    # translations
                            1, 0,      # a11, a12
                            0, 1,      # a21, a22
                            initial_cx, initial_cy]  # center
    else:  # 3D mode
        initial_cz = np.mean(source[:, 2])
        if transform_type == 'helmert':
            initial_params = [0, 0, 0,     # translations
                            0,             # scale
                            0, 0, 0,       # rotations
                            initial_cx, initial_cy, initial_cz]  # center
        else:  # affine
            initial_params = [0, 0, 0,     # translations
                            1, 0, 0,       # a11, a12, a13
                            0, 1, 0,       # a21, a22, a23
                            0, 0, 1,       # a31, a32, a33
                            initial_cx, initial_cy, initial_cz]  # center
    
    # Perform least squares optimization
    result = least_squares(residuals, initial_params, 
                         args=(source, target, mode, transform_type))
    
    return result.x

def print_results(params, source_data, target_data, mode='3D', transform_type='helmert'):
    """Print transformation results and error analysis"""
    print(f"\nTransformation Mode: {mode}")
    print(f"Transformation Type: {transform_type.capitalize()}")
    print("\nTransformation parameters:")
    print("-" * 50)
    if mode == '2D':
        if transform_type == 'helmert':
            print(f"Translation [tx, ty]: [{params[0]:.6f}, {params[1]:.6f}]")
            print(f"Scale: {params[2]:.6f}")
            print(f"Rotation (radians): {params[3]:.6f}")
            print(f"Center [cx, cy]: [{params[4]:.6f}, {params[5]:.6f}]")
        else:  # affine
            print(f"Translation [tx, ty]: [{params[0]:.6f}, {params[1]:.6f}]")
            print("Affine matrix:")
            print(f"[{params[2]:.6f}  {params[3]:.6f}]")
            print(f"[{params[4]:.6f}  {params[5]:.6f}]")
            print(f"Center [cx, cy]: [{params[6]:.6f}, {params[7]:.6f}]")
    else:  # 3D
        if transform_type == 'helmert':
            print(f"Translation [tx, ty, tz]: [{params[0]:.6f}, {params[1]:.6f}, {params[2]:.6f}]")
            print(f"Scale: {params[3]:.6f}")
            print(f"Rotation angles [ω, φ, κ] (radians): [{params[4]:.6f}, {params[5]:.6f}, {params[6]:.6f}]")
            print(f"Center [cx, cy, cz]: [{params[7]:.6f}, {params[8]:.6f}, {params[9]:.6f}]")
        else:  # affine
            print(f"Translation [tx, ty, tz]: [{params[0]:.6f}, {params[1]:.6f}, {params[2]:.6f}]")
            print("Affine matrix:")
            print(f"[{params[3]:.6f}  {params[4]:.6f}  {params[5]:.6f}]")
            print(f"[{params[6]:.6f}  {params[7]:.6f}  {params[8]:.6f}]")
            print(f"[{params[9]:.6f}  {params[10]:.6f}  {params[11]:.6f}]")
            print(f"Center [cx, cy, cz]: [{params[12]:.6f}, {params[13]:.6f}, {params[14]:.6f}]")

    # Apply transformation to source data
    transformed_data = transform_points(params, source_data, mode, transform_type)

    # Calculate residuals and statistics
    residuals = target_data - transformed_data
    
    # Calculate RMS error for each dimension
    rms_x = np.sqrt(np.mean(residuals[:, 0]**2))
    rms_y = np.sqrt(np.mean(residuals[:, 1]**2))
    if mode == '3D':
        rms_z = np.sqrt(np.mean(residuals[:, 2]**2))
    
    # Calculate total RMS error
    if mode == '2D':
        rms_total = np.sqrt(np.mean(np.sum(residuals[:, :2]**2, axis=1)))
    else:
        rms_total = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
    
    # Calculate statistics for each dimension
    min_residuals = np.min(residuals, axis=0)
    max_residuals = np.max(residuals, axis=0)
    mean_residuals = np.mean(residuals, axis=0)
    std_residuals = np.std(residuals, axis=0)
    
    print("\nDetailed Error Analysis:")
    print("-" * 50)
    print("RMS Errors:")
    print(f"X: {rms_x:.6f}")
    print(f"Y: {rms_y:.6f}")
    if mode == '3D':
        print(f"Z: {rms_z:.6f}")
    print(f"Total {'3D' if mode == '3D' else '2D'}: {rms_total:.6f}")
    
    print("\nResidual Statistics:")
    print("-" * 50)
    print(f"Dimension    Min         Max         Mean        Std Dev")
    print(f"X:       {min_residuals[0]:10.6f} {max_residuals[0]:10.6f} {mean_residuals[0]:10.6f} {std_residuals[0]:10.6f}")
    print(f"Y:       {min_residuals[1]:10.6f} {max_residuals[1]:10.6f} {mean_residuals[1]:10.6f} {std_residuals[1]:10.6f}")
    if mode == '3D':
        print(f"Z:       {min_residuals[2]:10.6f} {max_residuals[2]:10.6f} {mean_residuals[2]:10.6f} {std_residuals[2]:10.6f}")

    # Print sample results
    print("\nSample results (first 5 points):")
    print("Original | Transformed | Target")
    for i in range(5):
        original = source_data[i]
        transformed = transformed_data[i]
        target = target_data[i]
        print(f"{original[0]:.3f}, {original[1]:.3f}, {original[2]:.3f} | "
              f"{transformed[0]:.3f}, {transformed[1]:.3f}, {transformed[2]:.3f} | "
              f"{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}")

def get_file_path(prompt):
    """Get file path from user with validation"""
    while True:
        file_path = input(prompt).strip()
        try:
            with open(file_path, 'r') as f:
                # Test if file can be opened
                pass
            return file_path
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Please enter a valid file path.")
        except PermissionError:
            print(f"Error: Permission denied to access '{file_path}'. Please check file permissions.")
        except Exception as e:
            print(f"Error accessing file: {str(e)}")

def set_file_path(prompt):
    """Set new file path from user with validation"""
    while True:
        file_path = input(prompt).strip()
        try:
            with open(file_path, 'w') as f:
                # Test if file can be opened
                pass
            return file_path
        except PermissionError:
            print(f"Error: Permission denied to access '{file_path}'. Please check file permissions.")
        except Exception as e:
            print(f"Error accessing file: {str(e)}")

def save_transformation_params(params, mode, transform_type, output_file):
    """
    Save transformation parameters to a file in machine-readable format
    
    Args:
        params: transformation parameters array
        mode: '2D' or '3D'
        transform_type: 'helmert' or 'affine'
        output_file: path to output file
        
    Format:
        One parameter per line in key=value format
        All numerical values use 6 decimal places
    """
    try:
        with open(output_file, 'w') as f:
            f.write(f"mode={mode}\n")
            f.write(f"type={transform_type}\n")
            
            if mode == '2D':
                if transform_type == 'helmert':
                    f.write(f"tx={params[0]:.6f}\n")
                    f.write(f"ty={params[1]:.6f}\n")
                    f.write(f"scale={params[2]:.6f}\n")
                    f.write(f"rotation={params[3]:.6f}\n")
                    f.write(f"cx={params[4]:.6f}\n")
                    f.write(f"cy={params[5]:.6f}\n")
                else:  # affine
                    f.write(f"tx={params[0]:.6f}\n")
                    f.write(f"ty={params[1]:.6f}\n")
                    f.write(f"a11={params[2]:.6f}\n")
                    f.write(f"a12={params[3]:.6f}\n")
                    f.write(f"a21={params[4]:.6f}\n")
                    f.write(f"a22={params[5]:.6f}\n")
                    f.write(f"cx={params[6]:.6f}\n")
                    f.write(f"cy={params[7]:.6f}\n")
            else:  # 3D
                if transform_type == 'helmert':
                    f.write(f"tx={params[0]:.6f}\n")
                    f.write(f"ty={params[1]:.6f}\n")
                    f.write(f"tz={params[2]:.6f}\n")
                    f.write(f"scale={params[3]:.6f}\n")
                    f.write(f"omega={params[4]:.6f}\n")
                    f.write(f"phi={params[5]:.6f}\n")
                    f.write(f"kappa={params[6]:.6f}\n")
                    f.write(f"cx={params[7]:.6f}\n")
                    f.write(f"cy={params[8]:.6f}\n")
                    f.write(f"cz={params[9]:.6f}\n")
                else:  # affine
                    f.write(f"tx={params[0]:.6f}\n")
                    f.write(f"ty={params[1]:.6f}\n")
                    f.write(f"tz={params[2]:.6f}\n")
                    f.write(f"a11={params[3]:.6f}\n")
                    f.write(f"a12={params[4]:.6f}\n")
                    f.write(f"a13={params[5]:.6f}\n")
                    f.write(f"a21={params[6]:.6f}\n")
                    f.write(f"a22={params[7]:.6f}\n")
                    f.write(f"a23={params[8]:.6f}\n")
                    f.write(f"a31={params[9]:.6f}\n")
                    f.write(f"a32={params[10]:.6f}\n")
                    f.write(f"a33={params[11]:.6f}\n")
                    f.write(f"cx={params[12]:.6f}\n")
                    f.write(f"cy={params[13]:.6f}\n")
                    f.write(f"cz={params[14]:.6f}\n")
                    
        print(f"\nTransformation parameters saved to: {output_file}")
        
    except IOError as e:
        print(f"Error saving parameters to file: {str(e)}")
        raise


def main():
    print("\nCoordinate Transformation Tool")
    print("=" * 30)
    
    # Get file paths from user
    print("\nPlease provide the paths to your CSV files.")
    print("CSV files should have columns: ID, X, Y, Z, Type")
    source_file = get_file_path("\nEnter path to source coordinates CSV: ")
    target_file = get_file_path("Enter path to target coordinates CSV: ")
    
    # Read the CSV data
    try:
        source_data = read_csv(source_file)
        target_data = read_csv(target_file)
        
        # Validate data dimensions match
        if len(source_data) != len(target_data):
            raise ValueError("Source and target files have different numbers of points")
            
        # Allow user to choose transformation mode
        while True:
            mode = input("Choose transformation mode (2D/3D): ").strip().upper()
            if mode in ['2D', '3D']:
                break
            print("Invalid choice. Please enter '2D' or '3D'.")
        
        # Allow user to choose transformation type
        while True:
            transform_type = input("Choose transformation type (helmert/affine): ").strip().lower()
            if transform_type in ['helmert', 'affine']:
                break
            print("Invalid choice. Please enter 'helmert' or 'affine'.")
        
        # Compute transformation parameters
        params = compute_transformation(source_data, target_data, mode, transform_type)
        
        # Print results and error analysis
        print_results(params, source_data, target_data, mode, transform_type)
        
    except ValueError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please ensure your CSV files are properly formatted and contain valid coordinate data.")
    
       
    # Allow user to choose transformation mode
    while True:
        opt = input("\nDo you want to save the transformation parameters (Y/N)? ").strip().upper()
        if opt in ['Y', 'N']:
            break
        print("Invalid choice. Please enter 'Y' or 'N'.")

    if (opt == 'Y'):
        param_file = set_file_path("Enter path to transformation parameters file: ")   
        save_transformation_params(params, mode, transform_type, param_file)

if __name__ == "__main__":
    main()