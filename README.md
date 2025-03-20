# Physics-Informed Neural Networks (PINNs) for Simulating the LLG Equation

This repository contains the implementation of **Physics-Informed Neural Networks (PINNs)** to simulate the **Landau-Lifshitz-Gilbert (LLG) equation**. The LLG equation describes the dynamics of magnetization in ferromagnetic materials and is widely used in spintronics and magnonics research.

## Overview
The project explores the use of PINNs to solve the LLG equation by incorporating physical laws into the loss function of a deep learning model. The neural network is trained to approximate magnetization dynamics while ensuring consistency with the governing PDE.

### **Key Highlights**
- **Multi-Layer Feed-Forward Neural Networks (MLFFNN)** and **Recurrent Neural Networks (LSTMs)** are used.
- **Loss function** includes physical constraints such as PDE residuals and magnetization normalization.
- **Comparison of PINNs with traditional numerical solvers (ODE solvers) for LLG simulations**.
- **Evaluation of PINNs' effectiveness for multi-spin systems with interactions**.

## Methodology

1. **Defining the Problem:**  
   - The Landau-Lifshitz-Gilbert (LLG) equation is solved using neural networks.  
   - The input to the model is time, and the output is the three components of magnetization \((m_x, m_y, m_z)\).  
   - The model is trained to approximate the evolution of magnetization while ensuring it adheres to the LLG equation.  

2. **Loss Function Design:**  
   - **Mean Squared Error (MSE) on training data:** Ensures the model's predictions match the known solutions from numerical solvers.  
   - **Physics residual loss:** Penalizes deviations from the LLG equation by including the PDE residual in the loss function.  
   - **Normalization loss:** Enforces the physical constraint \(|m| = 1\), ensuring magnetization remains properly scaled.  

3. **Network Architectures:**  
   - **Multi-Layer Feed-Forward Neural Networks (MLFFNN):** Standard deep learning model with multiple layers.  
   - **Long Short-Term Memory (LSTM) Recurrent Neural Networks:** Captures time-dependent patterns in the magnetization dynamics.  

4. **Training & Evaluation:**  
   - Models are trained using the **Adaptive Moments (Adam) optimizer**.
   - Hyperparameters such as the damping parameter \(alpha\), number of collocation points, and training data size are optimized.  
   - The models are compared against traditional numerical solvers to evaluate accuracy and efficiency.  

---

## Results & Observations

- **PINNs successfully model magnetization dynamics** but require careful tuning of hyperparameters.  
- **Higher damping values (\(alpha\)) lead to better convergence** in PINNs, while lower values introduce numerical instability.  
- **LSTM-based models outperform feed-forward networks** in multi-spin simulations due to their ability to capture sequential dependencies.  
- **PINNs require significantly more computation time** than traditional ODE solvers but offer flexibility in handling irregular geometries and boundary conditions.  
- **Performance degrades for high-dimensional spin networks**, requiring better optimization strategies.  

---

## Future Work

- **Optimize PINN architectures** to improve computational efficiency and training time.  
- **Explore alternative loss functions** that better enforce the LLG equation constraints while reducing training instability.  
- **Extend simulations to larger multi-spin systems** by using grid-based neural networks for improved scalability.  
- **Combine PINNs with traditional solvers** to leverage the strengths of both data-driven and numerical approaches.  
- **Investigate Transformer-based architectures** for better handling of long-range spin interactions in large-scale simulations.  
