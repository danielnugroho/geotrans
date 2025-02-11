@echo off

REM Create a new Conda environment named 'GEOTRANS'
conda create -n GEOTRANS python=3.10 -y

REM Activate the environment
conda activate GEOTRANS

REM Install required packages
conda install -c conda-forge numpy scipy pandas -y
conda install -c conda-forge pdal -y
conda install -c conda-forge ezdxf -y
conda install -c conda-forge rasterio -y
conda install -c conda-forge tqdm -y
conda install -c conda-forge spyder -y


REM Install additional packages via pip
pip install pathlib
pip install ray

REM Print success message
echo Environment setup complete. Activate it using:
echo conda activate coordinate_transformation