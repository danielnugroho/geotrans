Coordinate Transformation Suite
============================

A GUI application for coordinate transformation operations, supporting multiple file formats and transformation types.

Features
--------

+ Transformation parameter computation from source/target coordinate pairs
+ Support for multiple file formats:
  - CSV (coordinate data)
  - LAZ/LAS (point clouds)
  - DXF (CAD files)
  - GeoTIFF (raster data)
+ Transformation types:
  - Helmert (similarity) transformation
  - Affine transformation
+ 2D and 3D transformation support
+ Parallel processing for large datasets
+ Real-time processing feedback
+ Detailed error analysis and statistics

Requirements
-----------

+ Python 3.10 or higher
+ Required packages:
  - tkinter (GUI components)
  - numpy (numerical computations)
  - scipy (optimization)
  - pdal (point cloud processing)
  - ezdxf (CAD file handling)
  - rasterio (GeoTIFF processing)
  - tqdm (progress bars)

Installation
------------

1. Create a new conda environment:

    conda create -n "geotrans" python=3.11
    conda activate geotrans

2. Install PDAL using conda-forge:

    conda install -c conda-forge pdal

3. Install other dependencies:

    pip install numpy scipy pandas ezdxf rasterio tqdm

Usage
-----

GUI Application:

1. Launch the application:

    python coordinate_transformation_tool.py

2. Parameter Computation:
   - Select source and target CSV files
   - Choose transformation mode (2D/3D)
   - Choose transformation type (Helmert/Affine)
   - Compute parameters
   - Save parameters to file

3. File Transformation:
   - Select input file to transform
   - Specify output file location
   - Load transformation parameters
   - Execute transformation

Input File Formats
-----------------

1. CSV files must have columns:
   - ID: Point identifier
   - X: X coordinate
   - Y: Y coordinate
   - Z: Z coordinate
   - Type: Point type (optional)

2. Supported file types:
   - CSV: Coordinate data
   - LAZ/LAS: Point cloud data
   - DXF: CAD files
   - GeoTIFF: Raster data

Transformation Types
-------------------

Helmert (Similarity) Transformation:
+ 2D: 4 parameters (2 translations, 1 scale, 1 rotation)
+ 3D: 7 parameters (3 translations, 1 scale, 3 rotations)
+ Preserves angles and shape
+ Minimum points required: 2 for 2D, 3 for 3D

Affine Transformation:
+ 2D: 6 parameters (2 translations, 4 transformation coefficients)
+ 3D: 12 parameters (3 translations, 9 transformation coefficients)
+ Allows for different scales in different directions
+ Minimum points required: 3 for 2D, 4 for 3D

Building Executable
------------------

Create standalone executable using PyInstaller:

    pyinstaller --onefile --clean --add-binary mkl_intel_thread.2.dll:. coordinate_transformation_tool.py

License
-------

This program is licensed under the GNU General Public License v3.0 (GPL-3.0).

Author
------

Daniel Adi Nugroho
Email: dnugroho@gmail.com

Version History
--------------

1.1.2 (2025-02-16)
+ Prepopulate processing output text pane with copyright and license information
+ Tested with PyInstaller and PyInstaller-compiled EXE works as expected
+ Works on any computer without Python installed

1.1.0 (2025-02-12)
+ First fully functional version with parallel processing
+ Source code cleanup
+ Modified to GNU-GPL license

1.0.0 (2025-02-08)
+ Initial release
+ GUI wrapper for geospatial data transformation functions