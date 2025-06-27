# 🧊 Room Cooling Simulation Prediction Using Neural Networks

## Overview

This project focuses on predicting the final physical state of a simulated room subjected to airflow and cooling dynamics using an artificial neural network (ANN). The simulations are designed to model how an air conditioning (AC) system and obstacles affect temperature, pressure, and velocity fields in a closed cubic space.

---

## 🧱 Simulation Setup

- **Room Geometry:** A cube of size 10 units per side, discretized with a **32×32×32** grid.

- **Initial Conditions:**
  - **Temperature (T):** Uniformly 25°C throughout the cube.
  - **Pressure (p):** Initially arbitrary (not relevant at t=0).
  - **Velocity (vx, vy, vz):** Zero everywhere at the start.

- **Dynamic Elements Introduced at t > 0:**
  - **Obstacles:** Regions where all velocity components (vx, vy, vz) remain zero (no airflow).
  - **Air Conditioners (ACs):**
    - Have predefined, non-zero velocity vectors.
    - Maintain a constant internal temperature of **20°C**.
    - Responsible for initiating and distributing airflow.

- **Simulation Outputs:**  
  The evolving state of the room is stored in VTK files, with per-grid-point values for:
  - Temperature (**T**)
  - Pressure (**p**)
  - Velocity components (**vx, vy, vz**)

---

## 🎯 Objective

To train an artificial neural network that can predict the final steady-state field values (**T, p, vx, vy, vz**) of the room given only the initial configuration at `t > 0`, including:
- Obstacle layout  
- AC positions and airflow characteristics (velocity vectors and fixed AC temperature)

---

## 📚 Dataset

- **Total Simulations:** 50  
- **Data Format:** VTK files containing full 3D fields at multiple timesteps  
- **Input:** Configuration of obstacles and ACs at early `t > 0`  
- **Output:** Final-time step fields of **T, p, vx, vy, vz**

---

## 🚀 Goal

This project aims to develop a fast surrogate model that replaces costly CFD-style simulations by learning the nonlinear dynamics of room cooling and airflow from data. The trained model can be used for:
- Fast design iteration of HVAC layouts  
- Inverse design and control  
- Real-time simulation and prediction

---

## 🧠 Learning Approach

This project leverages 3D Convolutional Neural Networks (3D-CNNs) to learn the physical evolution of the room. It processes voxelized data from VTK files and trains a model to forecast the final room state from the initial conditions and configuration.

---

## 🏗️ Project Structure (In progress)

```
├── data/                 # Raw and processed VTK files
├── models/               # CNN architecture and training scripts
├── utils/                # Preprocessing, VTK readers, etc.
├── outputs/              # Trained models and logs
├── notebooks/            # Jupyter notebooks for exploration and visualization
└── main.py               # Main training/evaluation script
```

---

## ⚙️ How to Run  (In progress)

1. **Install dependencies**  
   to be done
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

 
