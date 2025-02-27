# -*- coding: utf-8 -*-

__version__ = "1.2.0"
__author__ = "Daniel Adi Nugroho"
__email__ = "dnugroho@gmail.com"
__status__ = "Production"
__date__ = "2025-02-27"
__copyright__ = "Copyright (c) 2025 Daniel Adi Nugroho"
__license__ = "GNU General Public License v3.0 (GPL-3.0)"

# Version History
# --------------

# 1.2.0 (2025-02-27)
# - Added batch transformation tab for processing multiple files functionality

# 1.1.2 (2025-02-16)
# - Prepopulate processing output text pane with copyright and license information
# - Tested with Pyinstaller and PyInstaller-compiled EXE works as expected
# - Works on any computer without Python installed

# 1.1.0 (2025-02-12)
# - First fully functional version, with parallel processing implemented
# - Source code cleanup and modification to GNU-GPL license instead of MIT

# 1.0.0 (2025-02-08)
# - Initial release
# - GUI wrapper for geospatial data transformation functions

"""
Coordinate Transformation Suite GUI
=================================

A graphical user interface for coordinate transformation operations, providing tools for both
parameter computation and file transformation. This application serves as the main interface
for the coordinate transformation toolkit.

Purpose:
--------
- Provide user-friendly interface for coordinate transformation operations
- Enable parameter computation from source/target coordinate pairs
- Support file transformation using computed parameters
- Handle multiple file formats in a unified interface
- Provide real-time feedback and progress monitoring

Requirements:
------------
- Python 3.10 or higher
- Required packages:
  - tkinter: for GUI components
  - numpy: for numerical computations
  - get_localization_params.py: for parameter computation
  - transform_data.py: for file transformation operations
  - pathlib: for path handling
  - threading: for non-blocking operations
  - gc: for memory management

Features:
---------
1. Parameter Computation Tab:
   - Source and target CSV file selection
   - 2D/3D transformation mode selection
   - Helmert/Affine transformation type selection
   - Parameter computation and saving functionality
   - Real-time computation feedback

2. File Transformation Tab:
   - Support for multiple file formats:
     * CSV (coordinate data)
     * LAZ/LAS (point clouds)
     * DXF (CAD files)
     * GeoTIFF (raster data)
   - Input/output file selection
   - Parameter file loading
   - Progress monitoring

3. Console Output:
   - Real-time processing feedback
   - Error messages and warnings
   - Operation status updates

Usage:
------
1. Launch the application:
   python bce_coordinate_transformation_tool.py

2. Parameter Computation:
   - Select source and target CSV files
   - Choose transformation mode and type
   - Compute parameters
   - Save parameters to file

3. File Transformation:
   - Select input file to transform
   - Specify output file location
   - Load transformation parameters
   - Execute transformation

Notes:
------
- All file operations are performed in separate threads to prevent UI freezing
- Comprehensive error handling and user feedback
- Automatic cleanup of resources on application close
- Support for all major coordinate transformation types



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
print("Core imports starts.")
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
print("Tkinter imports finished.")
import sys, os, gc, datetime
import platform, subprocess
print("System imports finished.")
from pathlib import Path
print("Pathlib imports finished.")
import threading, multiprocessing
print("Multithreading/mp imports finished.")
print("Core imports finished.")

from get_localization_params import read_csv, compute_transformation, save_transformation_params, print_results
print("Localization params imports finished.")
from transform_data import read_params, transform_csv_data, transform_pointcloud, transform_dxf, transform_geotiff
print("Transform data imports finished.")

# this is for rasterio pyinstaller compatibility
print("Pyinstaler compatibility rasterio imports starts.")
import rasterio
import rasterio.control
import rasterio.crs
import rasterio.sample
import rasterio.vrt
import rasterio._features
print("All imports finished.")

class TransformationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Coordinate Transformation Suite")
        self.master.geometry("1024x600")
        
        # Initialize variables
        self.source_file = tk.StringVar()
        self.target_file = tk.StringVar()
        self.mode = tk.StringVar(value='3D')
        self.transform_type = tk.StringVar(value='helmert')
        self.params = None
        self.transform_input_file = tk.StringVar()
        self.transform_output_file = tk.StringVar()
        self.param_file = tk.StringVar()
        self.batch_source_folder = tk.StringVar()
        self.batch_target_folder = tk.StringVar()
        
        self.create_widgets()
        self.setup_output_redirect()
        
        # Add cleanup protocol
        self.master.protocol("WM_DELETE_WINDOW", self.cleanup)
        self._shutdown = False
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def create_widgets(self):
        # Create main notebook (tabs)
        notebook = ttk.Notebook(self.master)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Parameter Computation Tab
        param_frame = ttk.Frame(notebook)
        notebook.add(param_frame, text='Parameter Computation')
        self.create_param_computation_ui(param_frame)
        
        # File Transformation Tab
        transform_frame = ttk.Frame(notebook)
        notebook.add(transform_frame, text='File Transformation')
        self.create_file_transform_ui(transform_frame)

        # Batch Transformation Tab        
        batch_transform_frame = ttk.Frame(notebook)
        notebook.add(batch_transform_frame, text='Batch Transformation')
        self.create_batch_transform_ui(batch_transform_frame)

        # Console Output
        console_frame = ttk.LabelFrame(self.master, text="Processing Output")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5, anchor='s')
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=15)
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Display system information
        self.display_system_info()

    def create_param_computation_ui(self, parent):
        # File Selection Section
        file_frame = ttk.LabelFrame(parent, text="Coordinate Files")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Source CSV:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.source_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_source).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Target CSV:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.target_file, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_target).grid(row=1, column=2)
        
        # Options Section
        options_frame = ttk.LabelFrame(parent, text="Transformation Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(options_frame, text="Mode:").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(options_frame, text="2D", variable=self.mode, value='2D').grid(row=0, column=1)
        ttk.Radiobutton(options_frame, text="3D", variable=self.mode, value='3D').grid(row=0, column=2)
        
        ttk.Label(options_frame, text="Type:").grid(row=0, column=3, padx=5)
        ttk.Radiobutton(options_frame, text="Helmert", variable=self.transform_type, value='helmert').grid(row=0, column=4)
        ttk.Radiobutton(options_frame, text="Affine", variable=self.transform_type, value='affine').grid(row=0, column=5)
        
        # Control Buttons

        control_frame = ttk.Frame(parent)
        control_frame.pack(pady=10)
        
        self.compute_button = ttk.Button(control_frame, text="Compute Parameters", 
                                       command=self.run_parameter_computation)
        self.compute_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="Save Parameters", 
                                    command=self.save_parameters, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)        
        
    def create_file_transform_ui(self, parent):
        # File Selection Section
        file_frame = ttk.LabelFrame(parent, text="Transform Files")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.transform_input_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_transform_input).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.transform_output_file, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_transform_output).grid(row=1, column=2)
        
        ttk.Label(file_frame, text="Parameter File:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.param_file, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_param_file).grid(row=2, column=2)
        
        # Control Buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Transform File", 
                 command=self.run_file_transformation).pack(side=tk.LEFT, padx=5)

    def create_batch_transform_ui(self, parent):
        # Folder Selection Section
        folder_frame = ttk.LabelFrame(parent, text="Transform Folders")
        folder_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(folder_frame, text="Source Folder:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.batch_source_folder, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_batch_source_folder).grid(row=0, column=2)
        
        ttk.Label(folder_frame, text="Target Folder:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.batch_target_folder, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_batch_target_folder).grid(row=1, column=2)
        
        ttk.Label(folder_frame, text="Parameter File:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.param_file, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_param_file).grid(row=2, column=2)
        
        # Control Buttons
        control_frame = ttk.Frame(parent)
        control_frame.pack(pady=10)
        
        self.batch_transform_button = ttk.Button(control_frame, text="Transform Folder", 
                                            command=self.run_batch_transformation)
        self.batch_transform_button.pack(side=tk.LEFT, padx=5)
        
    # Add these methods to the TransformationApp class
    def browse_batch_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.batch_source_folder.set(folder)

    def browse_batch_target_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.batch_target_folder.set(folder)

    def run_batch_transformation(self):
        source_folder = self.batch_source_folder.get()
        target_folder = self.batch_target_folder.get()
        param_file = self.param_file.get()

        if not source_folder or not target_folder or not param_file:
            messagebox.showerror("Validation Error", "All fields are required")
            return

        # Disable the transform button to prevent multiple clicks
        self.batch_transform_button.config(state=tk.DISABLED)

        # Run the batch transformation in a separate thread
        def batch_transformation_thread():
            try:
                # Read parameters
                params = read_params(param_file)
                print("Transformation parameters:")
                print(params)

                # Create output directory if needed
                Path(target_folder).mkdir(parents=True, exist_ok=True)

                # Get list of supported files in the source folder
                supported_extensions = ['.csv', '.las', '.laz', '.dxf', '.tif', '.tiff']
                files_to_process = [f for f in Path(source_folder).iterdir() if f.suffix.lower() in supported_extensions]

                if not files_to_process:
                    messagebox.showwarning("No Files", "No supported files found in the source folder")
                    return

                # Process each file sequentially
                for input_file in files_to_process:
                    output_file = Path(target_folder) / input_file.name
                    ext = input_file.suffix.lower()

                    try:
                        if ext == '.csv':
                            transform_csv_data(str(input_file), str(output_file), params)
                        elif ext == '.dxf':
                            transform_dxf(str(input_file), str(output_file), params)
                        elif ext in ['.tif', '.tiff']:
                            transform_geotiff(str(input_file), str(output_file), params)
                        elif ext in ['.las', '.laz']:
                            transform_pointcloud(str(input_file), str(output_file), params)

                        print(f"Successfully transformed: {input_file.name}")
                    except Exception as e:
                        print(f"Error transforming {input_file.name}: {str(e)}")

                messagebox.showinfo("Success", "Batch transformation completed successfully!")
            except Exception as e:
                messagebox.showerror("Batch Transformation Error", str(e))
            finally:
                # Re-enable the transform button after the process is done
                self.batch_transform_button.config(state=tk.NORMAL)

        # Start the batch transformation in a new thread
        threading.Thread(target=batch_transformation_thread, daemon=True).start()

    def setup_output_redirect(self):
        sys.stdout = self
        sys.stderr = self
        
    def write(self, text):
        self.console_text.insert(tk.END, text)
        self.console_text.see(tk.END)
        
    def flush(self):
        pass
        
    def browse_source(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.source_file.set(filename)
            
    def browse_target(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.target_file.set(filename)
            
    def browse_transform_input(self):
        filename = filedialog.askopenfilename(filetypes=[
            ("Supported files", "*.csv *.las *.laz *.dxf *.tif *.tiff")
        ])
        if filename:
            self.transform_input_file.set(filename)
            
    def browse_transform_output(self):
        filename = filedialog.asksaveasfilename(filetypes=[
            ("Supported files", "*.csv *.las *.laz *.dxf *.tif *.tiff")
        ])
        if filename:
            self.transform_output_file.set(filename)
            
    def browse_param_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filename:
            self.param_file.set(filename)
            
    def validate_file_extensions(self, input_file, output_file):
        input_ext = os.path.splitext(input_file)[1].lower()
        output_ext = os.path.splitext(output_file)[1].lower()
        valid_extensions = {'.csv', '.laz', '.las', '.dxf', '.tif', '.tiff'}
        
        if input_ext not in valid_extensions:
            raise ValueError(f"Unsupported input file type: {input_ext}")
        if output_ext not in valid_extensions:
            raise ValueError(f"Unsupported output file type: {output_ext}")
        if input_ext != output_ext:
            raise ValueError("Input and output files must have the same extension")
            
    def run_parameter_computation(self):
        #self.console_text.delete(1.0, tk.END)
        try:
            source_data = read_csv(self.source_file.get())
            target_data = read_csv(self.target_file.get())
            
            if len(source_data) != len(target_data):
                raise ValueError("Source and target files have different numbers of points")
                
            params = compute_transformation(
                source_data, 
                target_data,
                self.mode.get(),
                self.transform_type.get()
            )
            
            self.params = params
            print("\nParameter computation successful!")
            print_results(params, source_data, target_data, self.mode.get(), self.transform_type.get())
            self.save_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Computation Error", str(e))
                       
    def save_parameters(self):
        """Handle parameter saving with file dialog"""
        # Modified check for numpy array existence
        if self.params is None or len(self.params) == 0:
            messagebox.showwarning("No Parameters", "Compute parameters first!")
            return
        
        # Get save path from user
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Transformation Parameters"
        )
        
        if not filename:  # User cancelled
            return
        
        try:
            save_transformation_params(
                self.params,
                self.mode.get(),
                self.transform_type.get(),
                filename
            )
            self.param_file.set(filename)  # Auto-set in transformation tab
            messagebox.showinfo("Success", f"Parameters saved successfully to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))        

    def run_file_transformation(self):
        #self.console_text.delete(1.0, tk.END) -- no need to clear console
        
        input_file = self.transform_input_file.get()
        output_file = self.transform_output_file.get()
        param_file = self.param_file.get()

        # Add shutdown check at start of critical methods
        if self._shutdown:
            messagebox.showwarning("Shutting Down", "Application is closing")
            return
        
        try:
            # Validate inputs
            if not input_file or not output_file or not param_file:
                raise ValueError("All file fields are required")
                
            self.validate_file_extensions(input_file, output_file)
            
            # Read parameters
            params = read_params(param_file)
            print("Transformation parameters:")
            print(params)
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_file)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
            # Run transformation in a thread to prevent UI freeze
            def transformation_thread():
                try:
                    ext = os.path.splitext(input_file)[1].lower()
                    
                    if ext == '.csv':
                        transform_csv_data(input_file, output_file, params)
                    elif ext == '.dxf':
                        transform_dxf(input_file, output_file, params)
                    elif ext in ['.tif', '.tiff']:
                        transform_geotiff(input_file, output_file, params)
                    elif ext in ['.las', '.laz']:
                        transform_pointcloud(input_file, output_file, params)
                        
                    messagebox.showinfo("Success", "Transformation completed successfully!")
                except Exception as e:
                    messagebox.showerror("Transformation Error", str(e))
                    
            threading.Thread(target=transformation_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))

    def cleanup(self):
        """Comprehensive resource cleanup method"""
        try:
            self._shutdown = True
            
            # 1. Restore original stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # 2. Stop any active threads
            for thread in threading.enumerate():
                if thread != threading.main_thread():
                    print(f"Terminating thread: {thread.name}")
                    # Add thread termination logic if needed
                    
            # 3. Clear Tkinter resources
            self.master.destroy()
            
            # 4. Force garbage collection
            gc.collect()
            
            # 5. Close any open file handles (example for rasterio)
            #if 'rasterio' in sys.modules:
            #    from rasterio._base import _active_interpreters
            #    _active_interpreters.clear()
                
            print("Cleanup completed successfully")
            
        except Exception as e:
            #messagebox.showerror("Cleanup Error", f"Error during cleanup: {str(e)}")
            print("Cleanup Error", f"Error during cleanup: {str(e)}")
        finally:
            # Use sys.exit for a graceful termination
            sys.exit(0)

    def __del__(self):
        """Destructor as final safety net"""
        self.cleanup()

    def get_system_specs(self):
        """
        Get detailed system specifications including CPU and RAM info.
        Returns a dictionary with system specifications.
        """
        specs = {}
        
        try:
            # First attempt: Try using psutil if available
            import psutil
            
            # CPU Info
            specs['processor'] = platform.processor()
            specs['cpu_cores'] = psutil.cpu_count(logical=True)
            specs['cpu_freq'] = f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "Unknown"
            
            # RAM Info (convert to GB and round to 2 decimal places)
            total_ram = psutil.virtual_memory().total
            specs['ram'] = f"{total_ram / (1024**3):.2f} GB"
            
        except ImportError:
            # Fallback for Windows: Use wmic commands
            if platform.system() == 'Windows':
                try:
                    # Get CPU info
                    cpu = subprocess.check_output('wmic cpu get name, maxclockspeed', shell=True).decode()
                    cpu_lines = cpu.strip().split('\n')[1:]  # Skip header
                    if cpu_lines:
                        specs['processor'] = cpu_lines[0].strip()
                    
                    # Get logical processors count
                    cpu_count = subprocess.check_output('wmic cpu get NumberOfLogicalProcessors', shell=True).decode()
                    count_lines = cpu_count.strip().split('\n')[1:]
                    if count_lines:
                        specs['cpu_cores'] = count_lines[0].strip()
                    
                    # Get RAM info
                    ram = subprocess.check_output('wmic computersystem get totalphysicalmemory', shell=True).decode()
                    ram_lines = ram.strip().split('\n')[1:]
                    if ram_lines:
                        ram_bytes = int(ram_lines[0].strip())
                        specs['ram'] = f"{ram_bytes / (1024**3):.2f} GB"
                        
                except:
                    # If wmic fails, use basic info
                    specs['processor'] = platform.processor()
                    specs['cpu_cores'] = "Unknown"
                    specs['ram'] = "Unknown"
            else:
                # For non-Windows systems, use basic info
                specs['processor'] = platform.processor()
                specs['cpu_cores'] = "Unknown"
                specs['ram'] = "Unknown"
        
        return specs

    def display_system_info(self):
        """
        Display system information in the console output panel when program starts.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get system specifications
        specs = self.get_system_specs()
        
        info_text = f"""
=== Coordinate Transformation Suite ===

Started on: {current_time}
Version: {__version__}
{__copyright__}
{__license__}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

System Information:
------------------
OS: {platform.system()} {platform.version()}
Architecture: {platform.machine()}
Python Version: {sys.version.split()[0]}
Running on: {platform.node()}

Hardware Information:
------------------
Processor: {specs.get('processor', 'Unknown')}
Logical Processors: {specs.get('cpu_cores', 'Unknown')}
Processor Speed: {specs.get('cpu_freq', 'Unknown')}
Total RAM: {specs.get('ram', 'Unknown')}

Ready for coordinate transformation tasks.
=======================================

"""
        # Insert the text into console
        self.console_text.insert(tk.END, info_text)
        self.console_text.see("1.0")

if __name__ == "__main__":
    print("MAIN PROGRAM STARTED")
    multiprocessing.freeze_support()
    print("MP freeze support placed.")
    root = tk.Tk()
    app = TransformationApp(root)
    root.mainloop()