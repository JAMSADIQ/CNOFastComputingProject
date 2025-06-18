import os
import importlib.util
import numpy as np
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt


def generate_vtk_and_plots(cfg, folder_path):
    shape = (cfg.res_x, cfg.res_y, cfg.res_z)
    voxel_size = (cfg.len_x / cfg.res_x, cfg.len_y / cfg.res_y, cfg.len_z / cfg.res_z)

    obstacle = np.zeros(shape, dtype=np.uint8)
    vx = np.zeros(shape, dtype=np.float32)
    vy = np.zeros(shape, dtype=np.float32)
    vz = np.zeros(shape, dtype=np.float32)

    def physical_to_index_range(coord_range, voxel_size, res):
        i_min = max(0, int(coord_range[0] / voxel_size))
        i_max = min(res, int(np.ceil(coord_range[1] / voxel_size)))
        return i_min, i_max

    def insert_rectangle(array, coords, value, voxel_size, shape):
        xi, xa = physical_to_index_range(coords[0], voxel_size[0], shape[0])
        yi, ya = physical_to_index_range(coords[1], voxel_size[1], shape[1])
        zi, za = physical_to_index_range(coords[2], voxel_size[2], shape[2])
        array[xi:xa, yi:ya, zi:za] = value

    def add_array_to_vtk(image, np_array, name, dtype):
        vtk_array = numpy_support.numpy_to_vtk(np_array.ravel(order='F'),
                                               deep=True,
                                               array_type=dtype)
        vtk_array.SetName(name)
        image.GetPointData().AddArray(vtk_array)

    # Insert obstacle
    for name, coords in cfg.obstacles_.items():
        insert_rectangle(obstacle, coords, value=1, voxel_size=voxel_size, shape=shape)

    # Insert inflow regions
    vx_val, vy_val, vz_val = cfg.ac_inflow[0] if isinstance(cfg.ac_inflow, list) else cfg.ac_inflow
    for name, coords in cfg.ac_.items():
        insert_rectangle(vx, coords, value=vx_val, voxel_size=voxel_size, shape=shape)
        insert_rectangle(vy, coords, value=vy_val, voxel_size=voxel_size, shape=shape)
        insert_rectangle(vz, coords, value=vz_val, voxel_size=voxel_size, shape=shape)

    # Create VTK file
    sp = vtk.vtkStructuredPoints()
    sp.SetDimensions(*shape)
    sp.SetSpacing(*voxel_size)
    sp.SetOrigin(0.0, 0.0, 0.0)

    add_array_to_vtk(sp, obstacle, "Obstacle", vtk.VTK_UNSIGNED_CHAR)
    add_array_to_vtk(sp, vx, "AC_vx", vtk.VTK_FLOAT)
    add_array_to_vtk(sp, vy, "AC_vy", vtk.VTK_FLOAT)
    add_array_to_vtk(sp, vz, "AC_vz", vtk.VTK_FLOAT)

    vtk_path = os.path.join(folder_path, "initial_config_file.vtk")
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(vtk_path)
    writer.SetInputData(sp)
    writer.Write()
    print(f"Saved: {vtk_path}")

    # Plot slices where vz ≠ 0
    z_slices = np.unique(np.where(vz != 0)[2])
    for z in z_slices:
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))
        axs[0].imshow(obstacle[:, :, z], origin='lower', cmap='gray')
        axs[0].set_title(f"Obstacle (Z={z})")

        im1 = axs[1].imshow(vx[:, :, z], origin='lower', cmap='coolwarm')
        axs[1].set_title("AC_vx")

        im2 = axs[2].imshow(vy[:, :, z], origin='lower', cmap='coolwarm')
        axs[2].set_title("AC_vy")

        im3 = axs[3].imshow(vz[:, :, z], origin='lower', cmap='coolwarm')
        axs[3].set_title("AC_vz")

        for ax, im in zip(axs[1:], [im1, im2, im3]):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plot_path = os.path.join(folder_path, f"preview_z_{z}.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Saved preview: {plot_path}")


# -------------------------------
# Loop over simulation folders
# -------------------------------
parent_dir = "./"#simulations"

for subfolder in ['1', '2']:#sorted(os.listdir(parent_dir)):
    folder_path = os.path.join(parent_dir, subfolder)
    if not os.path.isdir(folder_path):
        continue

    var_file_path = os.path.join(folder_path, "vars.py")
    if not os.path.exists(var_file_path):
        print(f"Skipping {folder_path}: no vars.py found")
        continue

    # Dynamic import of vars.py
    spec = importlib.util.spec_from_file_location("vars", var_file_path)
    vars_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vars_module)

    # Run generator
    generate_vtk_and_plots(vars_module, folder_path)

