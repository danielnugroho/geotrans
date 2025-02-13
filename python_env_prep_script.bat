@echo off
ECHO Create a new Conda environment named 'GEOTRANS-DEPLOY'
conda create -n GEOTRANS-DEPLOY python=3.10
ECHO Activate the environment
conda activate GEOTRANS-DEPLOY
ECHO Install required packages
conda install -c conda-forge numpy scipy pandas pdal ezdxf rasterio tqdm
ECHO Install additional packages via pip
pip install ray pathlib
ECHO Print success message
ECHO Environment setup complete. Activate it using:
ECHO conda activate GEOTRANS-DEPLOY