# reflect the data for each time of simualtion
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt

GRID_SHAPE = (32, 32, 32)

def read_vtk_file(filename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()

    data = reader.GetOutput()
    point_data = data.GetPointData()
    points = vtk_to_numpy(data.GetPoints().GetData())

    arrays = {
        'T': vtk_to_numpy(point_data.GetArray('T')),
        'p': vtk_to_numpy(point_data.GetArray('p')),
        'v': vtk_to_numpy(point_data.GetArray('v')),
        'points': points,
        'vtk_object': data
    }

    return arrays

def reshape_fields(arrays):
    T = arrays['T'].reshape(GRID_SHAPE)
    p = arrays['p'].reshape(GRID_SHAPE)
    v = arrays['v']
    vx = v[:, 0].reshape(GRID_SHAPE)
    vy = v[:, 1].reshape(GRID_SHAPE)
    vz = v[:, 2].reshape(GRID_SHAPE)
    return T, p, vx, vy, vz

def reflect_array(arr, axis, flip_sign=False):
    arr = np.flip(arr, axis=axis)
    return -arr if flip_sign else arr

def reflect_tensor_fields(T, p, vx, vy, vz, axis):
    T_new = reflect_array(T, axis)
    p_new = reflect_array(p, axis)
    vx_new = reflect_array(vx, axis, flip_sign=(axis == 0))
    vy_new = reflect_array(vy, axis, flip_sign=(axis == 1))
    vz_new = reflect_array(vz, axis, flip_sign=(axis == 2))
    return T_new, p_new, vx_new, vy_new, vz_new

def save_augmented_vtk(original_file, T, p, vx, vy, vz,  output_path):
    flat_T = T.flatten()
    flat_p = p.flatten()
    v_stack = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)

    # Use original grid structure
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(original_file)
    reader.Update()
    data = reader.GetOutput()

    # Create VTK arrays
    T_vtk = numpy_to_vtk(flat_T)
    T_vtk.SetName("T")
    p_vtk = numpy_to_vtk(flat_p)
    p_vtk.SetName("p")
    v_vtk = numpy_to_vtk(v_stack)
    v_vtk.SetName("v")

    # Assign data
    point_data = data.GetPointData()
    point_data.RemoveArray("T")
    point_data.RemoveArray("p")
    point_data.RemoveArray("v")
    point_data.AddArray(T_vtk)
    point_data.AddArray(p_vtk)
    point_data.AddArray(v_vtk)

    # Save file
    new_filename = original_file.replace("solution_", "solution_aug_")
    out_file = os.path.join(output_path, new_name)


    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(out_file)
    writer.SetInputData(data)
    writer.Write()
    print(f"Saved {new_filename}")

def plot_slices(T, p, vx, slice_index=16, title_prefix=""):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(vx[:, :, slice_index], cmap='seismic')
    plt.title(f"{title_prefix}Vx[:,:,{slice_index}]")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(T[:, :, slice_index], cmap='hot')
    plt.title(f"{title_prefix}T[:,:,{slice_index}]")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(p[:, :, slice_index], cmap='viridis')
    plt.title(f"{title_prefix}p[:,:,{slice_index}]")
    plt.colorbar()

    plt.tight_layout()
    #plt.savefig(f"{title_prefix}.png")
    #plt.show()

def process_vtk_file(filename, axis, output_dir):
    arrays = read_vtk_file(filename)
    T, p, vx, vy, vz = reshape_fields(arrays)

    plot_slices(T, p, vx, title_prefix="Original ")

    T_aug, p_aug, vx_aug, vy_aug, vz_aug = reflect_tensor_fields(T, p, vx, vy, vz, axis)

    plot_slices(T_aug, p_aug, vx_aug, title_prefix="Augmented ")
    plt.show()
    save_augmented_vtk(filename, T_aug, p_aug, vx_aug, vy_aug, vz_aug, output_dir)

# ----------- USAGE ------------
if __name__ == "__main__":
    base_dir = "simulations"
    axis = 0  # choose reflection axis: 0 = x, 1 = y, 2 = z

    for folder_num in range(1, 51):
        folder_name = str(folder_num)
        input_dir = os.path.join(base_dir, folder_name)
        output_dir = os.path.join(base_dir, f"{folder_name}aug")

        for i in range(10, 1001, 10):
            filename = f"solution_{i:04d}.vtk"
            full_path = os.path.join(input_dir, filename)

            if os.path.exists(full_path):
                process_vtk_file(full_path, axis, output_dir)
            else:
                print(f"Missing: {full_path}")
    #test one file            
    #folder_data = 'simulations'
    #simulation_names=
    #vtk_file = "1/solution_0010.vtk"  # adjust path
    #axis = 0  # reflection axis: 0 = x, 1 = y, 2 = z
    #process_vtk_file(vtk_file, axis)
Code
