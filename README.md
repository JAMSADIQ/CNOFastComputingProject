# CNOFastComputingProject
Multi-channel 3D-cube Successive Convolution Network 
# 3D Cube Evolution Predictor Using CNN

This project uses 3D Convolutional Neural Networks (3D-CNNs) to learn the time evolution of temperature and pressure in a cube. It reads voxelized data (e.g., VTK files), preprocesses the data into a consistent format, trains a neural network to predict future states from initial conditions, and evaluates generalization on unseen data.

## 🧠 Problem Statement

Given a cube of spatially varying **temperature** and **pressure** at time `t=0`, predict the fields at a future time `t=T` using a data-driven 3D CNN model. The network learns from a dataset of simulated evolutions over time.

## 📁 Project Structure


## 🚀 How to Run

1. **Install dependencies:**

   ```bash
   pip install .
