echo "Compiling Code...";

g++ -O3 -march=native -mtune=intel -msse4.2 -mavx2 -mfma -flto -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17 -I/usr/local/include/ fullVehicleSimulation.cpp -o fullVehicleSimulation;

echo "Compile Done, running code...";

./$1

python3 plotData.py