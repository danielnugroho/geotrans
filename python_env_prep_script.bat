@echo off
conda create -n "geoda" python=3.11
conda activate geoda
conda install -c conda-forge pdal
pip install ray numpy scipy pandas ezdxf rasterio tqdm pyinstaller
pyinstaller --onefile --noconsole --clean --add-binary mkl_intel_thread.2.dll:. coordinate_transformation_tool.py
rem We had a problem due to gdal installation which fails pyinstaller, hence gdal is not installed