# PINN for the 2D Lid-Driven Cavity Flow

## Introduction  

This repository is designed for solving the 2D steady incompressible lid-driven cavity flow at Reynolds number $R_e = 100$. The implemented algorithm corresponds to the Physics-Informed Neural Network (PINN) formulation, originally introduced by RAISSI M., PERDIKARIS P., and KARNIADAKIS G.E. (2019) in their paper *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"* (Journal of Computational Physics, 378, pp. 686–707, DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)).

A PINN is a deep learning model that incorporates the governing physical laws expressed as partial differential equations directly into the loss function in order to approximate the solution of the underlying physical system. Instead of relying on labeled data, the network is trained by minimizing the Navier–Stokes residuals and enforcing boundary conditions.

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Set Up the Python Environment**  

- To create and activate the virtual environment defined in `pyproject.toml` and `uv.lock`, execute the following command from the project root:

  ```bash
  uv sync
  source .venv/bin/activate  # On Windows use: .venv\Scripts\activate 
  ```

**2.** **Build and Train the PINN**  

- The `src/` folder contains the modular components to build and train the PINN:
  - `network.py`: Defines the neural architecture used to approximate the velocity and pressure fields. 
  - `pinn.py`: Implements the PINN, including the Navier–Stokes residuals and boundary condition enforcement.  
  - `optimizer.py`: Implements the optimization strategy, including learning rate, scheduler, and training parameters.
  - `main.py`: Trains the PINN and saves it to the `models/` directory.


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
