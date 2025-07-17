#python3 visualize_data.py --filename ~/Downloads/FastComputing/simulations2/1/solution_0000.vtk --plot  --zslice  2


import argparse
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

# Constants
GRID_SHAPE = (20, 20, 20)
DOMAIN_LENGTH = 10.0
STEP_SIZE = DOMAIN_LENGTH / GRID_SHAPE[0]  # 0.5

def read_vtk_file(filename):
    if filename.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif filename.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif filename.endswith('.vti'):
        reader = vtk.vtkXMLImageDataReader()
    elif filename.endswith('.vtk'):
        reader = vtk.vtkDataSetReader()
    else:
        raise ValueError("Unsupported VTK file format")

    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    point_data = data.GetPointData()
    point_arrays = {point_data.GetArrayName(i): vtk_to_numpy(point_data.GetArray(i))
                    for i in range(point_data.GetNumberOfArrays())}

    cell_data = data.GetCellData()
    cell_arrays = {cell_data.GetArrayName(i): vtk_to_numpy(cell_data.GetArray(i))
                   for i in range(cell_data.GetNumberOfArrays())}

    points = vtk_to_numpy(data.GetPoints().GetData())

    return {
        'points': points,
        'point_data': point_arrays,
        'cell_data': cell_arrays,
        'vtk_object': data
    }

def plot_stat_2d(stat_array, title):
    plt.figure()
    plt.imshow(stat_array, cmap='viridis')
    plt.title(title)
    plt.colorbar(label='Value')
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.tight_layout()
    plt.show()

def physical_to_index(physical_value):
    """Convert a domain unit value to index based on step size."""
    index = int(round(physical_value / STEP_SIZE))
    if not 0 <= index < GRID_SHAPE[2]:
        raise ValueError(f"Physical value {physical_value} is out of bounds (0 to {DOMAIN_LENGTH})")
    return index

def process_and_plot(data, zslice_val):
    temperature_values = data['point_data']['T']
    velocity_values = data['point_data']['v']
    Vx, Vy, Vz = velocity_values[:, 0], velocity_values[:, 1], velocity_values[:, 2]

    # Reshape to 20×20×20
    data_3d = temperature_values.reshape(GRID_SHAPE)

    mean_x, var_x = data_3d.mean(axis=0), data_3d.var(axis=0)
    mean_y, var_y = data_3d.mean(axis=1), data_3d.var(axis=1)
    mean_z, var_z = data_3d.mean(axis=2), data_3d.var(axis=2)

    # Plotting
    plot_stat_2d(mean_x, "Mean over X-axis (Y-Z view)")
    plot_stat_2d(mean_y, "Mean over Y-axis (X-Z view)")
    plot_stat_2d(mean_z, "Mean over Z-axis (X-Y view)")

    plot_stat_2d(var_x, "Variance over X-axis (Y-Z view)")
    plot_stat_2d(var_y, "Variance over Y-axis (X-Z view)")
    plot_stat_2d(var_z, "Variance over Z-axis (X-Y view)")

    # Mid-slice or user-specified slice
    index = physical_to_index(zslice_val)
    print(f"Plotting Z slice at domain position {zslice_val} (index {index})")
    plt.imshow(data_3d[:, :, index], cmap='viridis')
    plt.title(f"Z Slice at domain position {zslice_val} (index {index})")
    plt.colorbar(label="Temperature")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="VTK Plotter with Slice Selector")
    parser.add_argument("--filename", type=str, required=True, help="Path to VTK file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--zslice", type=float, default=5.0, help="Z-slice location in domain units (0 to 10)")

    args = parser.parse_args()
    data = read_vtk_file(args.filename)

    print(f"\nFile: {args.filename}")
    print(f"- {len(data['points'])} points")
    print("Point data arrays:", list(data['point_data'].keys()))
    print("Cell data arrays:", list(data['cell_data'].keys()))

    ##check min max values
    temperature_values = data['point_data']['T']
    velocity_values = data['point_data']['v']
    Vx, Vy, Vz = velocity_values[:, 0], velocity_values[:, 1], velocity_values[:, 2]

    print(f"min/max Vx: {min(Vx)} / {max(Vx)}")
    print(f"min/max Vy: {min(Vy)} / {max(Vy)}")
    print(f"min/max Vz: {min(Vz)} / {max(Vz)}")
    print(f"min/max T:  {min(temperature_values)} / {max(temperature_values)}")
    print(f"First 5 T values: {temperature_values[:5]}")

    if args.plot:
        process_and_plot(data, args.zslice)

if __name__ == "__main__":
    main()

