#python pyvista_plotter.py --filename solution_0000.vtk --scalar T --vector v --component y --slice_plane z --slice_pos 5.0

#python pyvista_plotter.py --filename solution_0000.vtk --scalar T --crop 2 8 2 8 2 8

import argparse
import pyvista as pv
import numpy as np


def load_data(filename):
    """Load VTK-compatible dataset using PyVista."""
    return pv.read(filename)


def plot_3d_scalar(data, scalar_field, bounds=None):
    """Plot a 3D scalar field (e.g., temperature) with optional cropping."""
    if bounds:
        data = data.clip_box(bounds, invert=False)

    plotter = pv.Plotter()
    plotter.add_volume(data, scalars=scalar_field, cmap="viridis", opacity="sigmoid")
    plotter.add_axes()
    plotter.add_title(f"3D Volume: {scalar_field}")
    plotter.show()


def plot_2d_slice(data, vector_field, component='x', plane='z', slice_position=5.0):
    """Plot 2D slice of one component of a vector field."""
    # Extract vector component
    if vector_field not in data.array_names:
        raise ValueError(f"Vector field '{vector_field}' not found.")

    vectors = data[vector_field]
    component_index = {'x': 0, 'y': 1, 'z': 2}[component.lower()]
    scalar_name = f"{vector_field}_{component}"

    # Add component as scalar array for plotting
    data[scalar_name] = vectors[:, component_index]

    # Take a slice
    sliced = data.slice(normal=plane, origin=data.center)
    sliced = sliced.slice(normal=plane, origin=(0, 0, slice_position))

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(sliced, scalars=scalar_name, cmap="coolwarm", show_edges=False)
    plotter.add_axes()
    plotter.add_title(f"2D Slice: {vector_field} {component.upper()} in {plane.upper()} plane at {slice_position}")
    plotter.show()


def main():
    parser = argparse.ArgumentParser(description="PyVista VTK Viewer")
    parser.add_argument("--filename", required=True, help="Path to VTK file")
    parser.add_argument("--scalar", default="T", help="Scalar field to render in 3D (default: T)")
    parser.add_argument("--vector", default="v", help="Vector field name (default: v)")
    parser.add_argument("--component", default="x", choices=["x", "y", "z"], help="Vector component to plot")
    parser.add_argument("--slice_plane", default="z", choices=["x", "y", "z"], help="Plane for slicing")
    parser.add_argument("--slice_pos", type=float, default=5.0, help="Position in domain for slicing (default: 5.0)")
    parser.add_argument("--crop", nargs=6, type=float, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                        help="Crop bounds in 3D space")
    args = parser.parse_args()

    data = load_data(args.filename)

    if args.scalar not in data.array_names:
        raise ValueError(f"Scalar field '{args.scalar}' not found in the file.")

    # Plot 3D scalar volume
    plot_3d_scalar(data, args.scalar, bounds=args.crop)

    # Plot 2D slice of vector component
    plot_2d_slice(data, args.vector, component=args.component, plane=args.slice_plane, slice_position=args.slice_pos)


if __name__ == "__main__":
    main()

