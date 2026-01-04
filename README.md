# PINN for the 2D Lid-Driven Cavity Flow

## Introduction  

This repository is designed for solving the 2D steady incompressible lid-driven cavity flow at Reynolds number $Re = 100$. The implemented methodology corresponds to the Physics-Informed Neural Network (PINN) formulation, originally introduced by RAISSI M., PERDIKARIS P., and KARNIADAKIS G.E. (2019) in their paper *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"* (Journal of Computational Physics, 378, pp. 686–707, DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)).

A PINN is a deep learning model that embeds the governing physical laws, expressed as partial differential equations, directly into the loss function to approximate the solution of the underlying physical system. Rather than relying on labeled data, the network is trained by minimizing the Navier–Stokes residuals and enforcing the boundary conditions.

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Set Up the Python Environment**  

- To create and activate the virtual environment defined in `pyproject.toml` and `uv.lock`, execute the following command:

  ```bash
  uv sync
  source .venv/bin/activate  # On Windows use: .venv\Scripts\activate 
  ```

**2.** **Run the PINN Implementation**  

- The `src/` folder contains the modular components of the PINN implementation:
  - `network.py`: Defines the neural architecture that maps spatial coordinates to the corresponding streamfunction and pressure fields. 
  - `pinn.py`: Implements the PINN, including the Navier–Stokes residuals and boundary condition enforcement.  
  - `optimizer.py`: Provides the training loop, performing loss evaluation, backpropagation, and parameter updates.
  - `main.py`: Runs the PINN implementation and stores the resulting model in the `models/` directory.

- Run the following command to execute the full workflow:

  ```bash
  python main.py
  ```

**3.** **Compute the FDM Numerical Solution**

- The `src/` folder contains the `fdm.py` module, which computes the finite‑difference approximation of the lid‑driven cavity flow using the vorticity–streamfunction formulation.

- Run the following command to execute the FDM computation:

  ```bash
  python fdm.py
  ```


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
