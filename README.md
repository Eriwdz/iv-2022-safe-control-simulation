# iv-2022-safe-control-simulation

Simulation Code for S. Gangadhar, Z. Wang, H. Jing and Y. Nakahira, "Adaptive Safe Control for Driving in Uncertain Environments," 2022 IEEE Intelligent Vehicles Symposium (IV), 2022, pp. 1662-1668, doi: 10.1109/IV51971.2022.9827264. 

Please refer to the "Vehicle_Dynamics_Model_Readme.pdf" for details on the vehicle model and how the simulation was setup to incorporate SIMD vectorization and other numerical stability components.

## Strcture 
* the project contains a .cpp file titled "fullVehicleSimulation" that contains all of the simulation code and a .py file titled "plotResults" that plots the simulation results.
* to run, simply execute the run.sh shell scipt.

## Development
**Requirements:**
* Ubuntu 20.04
* Eigen 3.4 installed using cmake and make, see [Eigen: Getting Started][https://eigen.tuxfamily.org/dox/GettingStarted.html]
* Python3 with numpy, matplotlib, and scipy