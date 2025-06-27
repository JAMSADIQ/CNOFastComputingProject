# CNOFastComputingProject  
Multi-channel 3D-Cube Successive Convolution Network  
# 🧠 3D Cube Evolution Predictor Using CNN

This project leverages 3D Convolutional Neural Networks (3D-CNNs) to predict the physical evolution of a simulated room (cube) under the influence of airflow and cooling introduced via air conditioners (ACs). It processes 3D voxelized simulation data stored in VTK files and trains a neural network to forecast final state fields from an initial setup.

---

## 📌 Problem Statement

Given a room (cube of size 10×10×10, discretized as 32×32×32 grid points) with the following setup at time `t > 0`:
- Spatial layout of **obstacles** (regions with zero velocity)
- **Air conditioners** (regions with fixed temperature = 20°C and given non-zero velocity)
- Initial **temperature**, **pressure**, and **velocity** fields

→ **Predict the final steady-state fields**:
- Temperature (**T**)
- Pressure (**p**)
- Velocity components (**vx, vy, vz**)

---

## 📂 Data Description

- **Simulations:** 50 CFD-style simulations
- **Input Format:** VTK files
- **Channels per voxel:**  
  - Temperature (**T**)  
  - Pressure (**p**)  
  - Velocity: **vx**, **vy**, **vz**
- **Initial Conditions:**  
  - T = 25°C throughout  
  - p is arbitrary at t=0  
  - vx = vy = vz = 0  
- **Obstacles:** Zones with all velocity components = 0  
- **AC Units:**  
  - T = 20°C inside  
  - Prescribed velocities (vx, vy, vz)

---

## 🏗️ Project Structure

```
├── data/                 # Raw and processed VTK files few ones
├── models/               # CNN architecture and training scripts
├── utils/                # Preprocessing, VTK readers, etc.
├── outputs/              # Trained models and logs
├── notebooks/            # Jupyter notebooks for exploration and visualization
└── main.py               # Main training/evaluation script
```

---

## 🚀 How to Run

1. **Install dependencies**  
   (Make sure you have Python ≥3.8)

   ```bash
   pip install .
   ```

2. **Train the model**

   ```bash
   python main.py --train
   ```

3. **Evaluate the model**

   ```bash
   python main.py --eval
   ```

---

## 🎯 Goal

The ultimate objective is to train a surrogate model capable of **predicting the final state** of the system quickly and accurately—replacing time-consuming physical simulations. This can enable:
- Fast HVAC layout testing  
- Inverse design for airflow scenarios  
- Real-time environmental control predictions

---
