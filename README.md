# iv-2022-safe-control-simulation

Simulation Code for S. Gangadhar, Z. Wang, H. Jing and Y. Nakahira, "Adaptive Safe Control for Driving in Uncertain Environments," 2022 IEEE Intelligent Vehicles Symposium (IV), 2022, pp. 1662-1668, doi: 10.1109/IV51971.2022.9827264. 

Please refer to the "Vehicle_Dynamics_Model_Readme.pdf" for details on the vehicle model and how the simulation was setup to incorporate SIMD vectorization and other numerical stability components.

## Strcture 
* the project root is a self-contained C++ code without a cmakelist
* the code makes use of Eigen, a C++ library of template headers for linear algebra
* main code is fullVehicleSimulation.cpp

## Development
**Requirements:**
* Ubuntu 20.04
* GCC is required with atleast c++17
* Eigen 3.4 installed using cmake and make, see [Eigen: Getting Started][https://eigen.tuxfamily.org/dox/GettingStarted.html]
* Python3 with numpy, matplotlib, and scipy

## How to run

* for Linux and Mac, run the following command: 
> . run.sh
