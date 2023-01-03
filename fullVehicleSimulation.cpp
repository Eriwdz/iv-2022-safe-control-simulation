/*
Created on Feb 15 23:56:34 2022
​
@author: siddg

This cpp file includes a detailed vehicle model with full-body tire model, balanced drivetrain, and free-body vehicle dynamics.
We use a sparse MPC as the baseline nominal controller, and apply a stochastic safety aware controller on top as an optimization problem. 
*/

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "eiquadprog.hpp"
#include "sparseModelPredictiveController.hpp"
#include <chrono>
#include <random>
#include <functional>

#define PI 3.1415926535

// Declare commonly used namespaces

using namespace Eigen;
using namespace std;

// Declare CSV read and write format and comma delimiter

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

// For storing reduced-order planned trajectory information (both state and control)

MatrixXf traj_data;

// Save and load CSV files into Eigen matrices

class CSVData
{
    public:
    MatrixXf data;
    string filename;

    CSVData(string filename_)
    {
        filename = filename_;
    }

    CSVData(string filename_, MatrixXf data_)
    {
        filename = filename_;
        data = data_;
    }

    void writeToCSVfile()
    {
        ofstream file(filename.c_str());
        file << data.format(CSVFormat);
        file.close();
    }

    MatrixXf readFromCSVfile()
    {
        vector<float> matrixEntries;
        ifstream matrixDataFile(filename);
        string matrixRowString;
        string matrixEntry;
        int matrixRowNumber = 0;
    
        while (getline(matrixDataFile, matrixRowString))
        {
            stringstream matrixRowStringStream(matrixRowString);
            while (getline(matrixRowStringStream, matrixEntry, ','))
            {
                matrixEntries.push_back(stod(matrixEntry));
            }
            matrixRowNumber++;
        }
        
        return Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    }
};

/*
This is the main vehicle dynamics class.
Dynamics are presented in control-affine form, i.e, x_dot = fs(x) + gs(x) u
*/

class VehicleDynamics
{
    public:
    // Vehicle parameters
    float m, lf, lr, l, B, Jw, Iz, R, b0, b1, Cl, Cs, T, g, Fz, c1, c2, c3, w1, w2;
    // These are used to speed-up and optimize simulation computations
    ArrayXf q1, q2, q3, q4, epi;

    VehicleDynamics()
    {
        
        m = 1500.00; // Vehicle mass
        
        lf = 1.070; // C.G. to front mid-axle
        
        lr = 1.605; // C.G. to read mid-axle
        
        l = lf + lr; // Vehicle base-length
        
        B = 1.517/2; // Front and rear axle width
        
        Iz = 2600.00; // Vehicle moment of inertia
        
        Jw = 5.00; // Tire moment of inertia
        
        R = 0.316; // Wheel radius

        b0 = 1.25; // Tire drag coefficient
        
        b1 = 5.00; // Aerodynamic drag coefficient
        
        g = 9.81; // Acceleration due to gravity
        
        Fz = m*g; // Vehicle weight

        // Tire adhesion coefficients (determined by the intrisics of the tire and the terrain)
        c1 = 0.86;
        c1 = 33.82; 
        c1 = 0.35;

        // Simulation variables to enable vectorization and eliminate loops 
        q1 = ArrayXf::Zero(4); q1 << -1, 1, -1, 1;
        q2 = ArrayXf::Zero(4); q2 << -1, 1, 1, -1;
        q3 = ArrayXf::Zero(4); q3 << 1, 1, 0, 0;
        q4 = ArrayXf::Zero(4); q4 << 0, 0, 1, 1;

        epi = ArrayXf::Constant(4, 1, 1e-4); // Discrete step needed for gradient computation
    }

    // Tire adhesion calculation
    ArrayXf mu(ArrayXf s)
    {
        return c1*(1 - exp(-c2*s)) - (c3)*s;
    }

    // These are smooth max and min functions, needed to prevent stiffness when solving the system dynamics. 
    // Using smooth approximations of discontinuous functions, in ODEs, leads to Filippov solutions, which are numerically stable (Lipschitz Continuity)
    ArrayXf smoothMax(ArrayXf x1, ArrayXf x2)
    {
        ArrayXf P = ArrayXf::Constant(x1.rows(), x1.cols(), 1e-3);
        return 0.50*(x1 + x2 + sqrt(square(x2 - x1) + P));
    }
    ArrayXf smoothMin(ArrayXf x1, ArrayXf x2)
    {
        ArrayXf P = ArrayXf::Constant(x1.rows(), x1.cols(), 1e-3);
        return -0.50*(-x1 - x2 + sqrt(square(x2 - x1) + P));
    }

    // This is a sigmoid function and works as a smooth approximations of the heaviside step-function
    ArrayXf expit(ArrayXf x)
    {
        return (tanh(25*x) + 1)*0.50;
    }

    // Here we compute the actual tire forces generated, given the vehicle's current state
    MatrixXf tireForces(VectorXf x)
    {
        MatrixXf S_stack(4, 2); // This matrix stacks the longitudinal and lateral slip ratios for each tire (4 x 2) - highly vectorized
        
        MatrixXf FTire(4, 2); // This matrix stacks the generated longitudinal and lateral forces for each tire (4 x 2) - highly vectorized
        
        float beta = lr/l*tan(x(10)); // This the vehicle slip angle, deterimed by the current steering angle x(10)
        
        ArrayXf vCOG = ArrayXf::Constant(4, 1, x(seq(3, 4)).matrix().norm()); // This is a (4 x 1) array that holds the vehicle's C.G. velocity - norm of the longitudinal and lateral velocities
        
        ArrayXf vR = x(seq(6, 9))*R; // This is a (4 x 1) array that stores the angular tire speeds - not the actual ground-point tire velocity
        
        ArrayXf vTire = vCOG + x(5)*(q1*B*cos(beta) - q1*q2*lf*sin(beta)); // This ia a (4 x 1) array that stores the actual ground-point tire velocities
        
        float alphaF = -beta + x(10) - lf*x(5)/vCOG(0); // This is the front tire slip angle - note that the front two tires have been lumped together
        
        float alphaR = -beta + lr*x(5)/vCOG(0); // This is the rear tire slip angle - note that the rear two tires have been lumped together
        
        ArrayXf Sl = (vR*(q3*cos(alphaF) + q4*cos(alphaR)) - vTire) / smoothMax(smoothMax(vR*(q3*cos(alphaF) + q4*cos(alphaR)), vTire), epi); // This is a (4 x 1) array of the longitudinal tire slip ratios - one for each tire
        
        ArrayXf Ss = (q3*tan(alphaF) + q4*tan(alphaR))*expit(Sl) + (q3*sin(alphaF) + q4*sin(alphaR))*(vR/vTire)*expit(-Sl); // This is a (4 x 1) array of the lateral tire slip ratios - one for each tire
        
        S_stack << Sl, Ss; // Stack these two arrays into a single matrix
        
        ArrayXf Sr = S_stack.rowwise().norm(); // Reduce the stacked (4 x 2) matrix into a (4 x 1) array of the 2-norm of the slip ratios
        
        ArrayXf Mu = mu(Sr); // Calulate the tire adhesion for each tire - (4 x 1) array

        // Compute the longitudinal and lateral tire forces in the tire's frame of reference - not in the vehicle's local reference
        ArrayXf Fl = Sl*Mu*Fz/Sr;
        ArrayXf Fs = Ss*Mu*Fz/Sr;

        // Apply rotational transformations to bring Fl and Fs to the vehicle's frame of reference
        FTire.row(0) = (Fl(seq(0, 1))*cos(alphaF) + Fs(seq(0, 1))*sin(alphaF))*cos(x(10)) - (Fs(seq(0, 1))*cos(alphaF) - Fl(seq(0, 1))*sin(alphaF))*sin(x(10));
        FTire.row(1) = (Fl(seq(0, 1))*cos(alphaF) + Fs(seq(0, 1))*sin(alphaF))*sin(x(10)) + (Fs(seq(0, 1))*cos(alphaF) - Fl(seq(0, 1))*sin(alphaF))*cos(x(10));
        FTire.row(2) = Fl(seq(2, 3))*cos(alphaR) + Fs(seq(2, 3))*sin(alphaR);
        FTire.row(3) = Fs(seq(2, 3))*cos(alphaR) - Fl(seq(2, 3))*sin(alphaR);
 
        return FTire; // Return the compute tire-forces
    }

    // Vehicle Dynamics - fs(x)
    VectorXf fs(VectorXf x)
    {
        
        float Fd = b1*x(3); // Aerodynamic drag force
        
        MatrixXf F = tireForces(x); // Compute all tire forces
        
        VectorXf x_dot(11); // Initialize Vector

        // Compute the actual function
        x_dot << x(3)*cos(x(2)) - x(4)*sin(x(2)),
                 x(3)*sin(x(2)) + x(4)*cos(x(2)),
                 x(3)/l*tan(x(10)),
                 x(4)*x(5) + 1/m*(F(0, 0) + F(0, 1) + F(2, 0) + F(2, 1)) - Fd,
                 -x(3)*x(5) + 1/m*(F(1, 0) + F(1, 1) + F(3, 0) + F(3, 1)),
                 1/Iz*( lf*(F(1, 0) + F(1, 1)) - lr*(F(3, 0) + F(3, 1)) + B*(F(2, 1) + F(0, 1) - F(2, 0) - F(0, 0)) ),
                 1/Jw*( -R*F(0, 0) - b0*x(6)),
                 1/Jw*( -R*F(0, 1) - b0*x(7)),
                 1/Jw*( -R*F(2, 0) - b0*x(8)),
                 1/Jw*( -R*F(2, 1) - b0*x(9)),
                 0;

        return x_dot;
    }

    // Vehicle Dynamics - gs(x)
    MatrixXf gs(VectorXf x)
    {
        MatrixXf G = MatrixXf::Zero(x.rows(), 2);
        G(3, 0) = cos(x(10)); G(4, 0) = sin(x(10));
        G(x.rows()-1, 1) = 1.0;
        return G;
    }

    // Complete Vehicle Dynamics - fs(x) + gs(x) u
    VectorXf f_sys(VectorXf x, VectorXf u)
    {
        return fs(x.array()) + gs(x.array())*u;
    }

    // MPC Controller
    VectorXf ctrl(float t, VectorXf x)
    {   
        
        int k = int(1000*t); // Discrete time step k (using a time-step of 0.001)
        
        int Nh = 200; // MPC time horizon
        
        VectorXf X_ref = traj_data(seq(0, 2), seq(k, k + Nh - 1)); // Planned State reference trajectory
        
        VectorXf U_ref = traj_data(seq(3, 4), seq(k, k + Nh - 1)); // Planned Control reference trajectory
        
        VectorXf U_mpc = mpc.solveMPC(X(seq(0, 2), i), X_ref, U_ref); // Obtain Control action from kinematic MPC (velocity and steering angle)

        // Pass control action back as an acceleration input using an intermediate proportional controller (convert velocity and steering angles into accelerations and steering rates)
        VectorXf u(2);
        u << 200*(U_mpc(0) - x(3)),
             2.5*(U_mpc(1) - x(10));
        
        return u;
    }

    // Final closed-loop dynamics x_dot = func(t, x) function that can be solved using RK4
    VectorXf func(float t, VectorXf x)
    {
        VectorXf u = ctrl(t, x);
        return f_sys(x, u);
    }
};


class StochasticCBF
{
    public:

    float Ts; // Discrete time step

    int N_horizon; // Safe controller preview horizon

    int N_episode; // Number of monte-carlo simulation episodes

    float epsilon = 0.10; // Finite-difference gradient step size

    MatrixXd Q, Ge, Gi; // Matrix variables for QP-solver

    VectorXd u0, he, hi, u; // Vector variables for QP-solver

    VehicleDynamics vd; // Instance of vehicle dynamics

    StochasticCBF(float Ts_init, int N_horizon_init, int N_episode_init)
    {
        Ts = Ts_init;

        N_horizon = N_horizon_init;

        N_episode = N_episode_init;

    }

    // Here, the smooth max and min functions are used for fuzzy implementations of boolean compositions
    // This guarantees consistent barrier function gradients.
    float smoothMax(float x1, float x2)
    {
        return 0.50*(x1 + x2 + sqrt((x2 - x1)*(x2 - x1) + 1e-4));
    }
    float smoothMin(float x1, float x2)
    {
        return -0.50*(-x1 - x2 + sqrt((x2 - x1)*(x2 - x1) + 1e-4));
    }

    // This is a wrapper function on top of the eiquadprog QP-solver
    // Typecasts incoming matrices and vectors into doubles for eiquadprog and converts them back before returning
    VectorXf solveQP(VectorXf u0_, MatrixXf Gi_, VectorXf hi_)
    {
        u0 = -1*u0_.cast <double> ();

        Gi = (Gi_.transpose()).cast <double> ();

        hi = hi_.cast <double> ();

        Q = MatrixXd::Identity(u0.rows(), u0.rows());

        Ge.resize(Q.rows(), 0);

        he.resize(0);

        solve_quadprog(Q, u0, Ge, he, Gi, hi, u); 

        VectorXf u_f = u.cast <float> ();

        return u_f;
    }

    // This is a random normal vector generator
    VectorXf randNormal(int n, float std)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // Pass the current system clock as a seed for the random generator

        default_random_engine generator(seed); // Create the random generator

        normal_distribution<float> distribution(0.0, std); // Create the normal distribution

        auto normal = [&] (float) {return distribution(generator);}; // Use a lambda expression to wrap the generator function so it can be passed as an argument

        VectorXf d = VectorXf::NullaryExpr(2, normal); // This is a nullary-expression that is used to define procedural matrices such as constant or random matrices

        VectorXf v = VectorXf::Zero(n, 1); 
        v(seq(3, 4)) = d; // Add noise to only a part of the state-vector
        return v;
    }

    // This is the barrier function (0-superlevel set definer) to specify the safety condition
    float phi(VectorXf x)
    {
        float eta = 0.85; // Percentage of maximum tire force

        float zeta = (eta*vd.m*vd.g)*(eta*vd.m*vd.g); // Square of limiting tire force (desired barrier)

        MatrixXf F = vd.tireForces(x); // Compute tire forces using current state estimate

        // Compute squared 2-norm of tire forces - one for each tire
        float Ffl_2 = F(0, 0)*F(0, 0) + F(1, 0)*F(1, 0);
        float Ffr_2 = F(0, 1)*F(0, 1) + F(1, 1)*F(1, 1);
        float Frl_2 = F(2, 0)*F(2, 0) + F(3, 0)*F(3, 0);
        float Frr_2 = F(2, 1)*F(2, 1) + F(3, 1)*F(3, 1);

        // Calculate the barrier function for each tire
        float hfl = 1 - (Ffl_2/zeta);
        float hfr = 1 - (Ffr_2/zeta);
        float hrl = 1 - (Frl_2/zeta);
        float hrr = 1 - (Frr_2/zeta);

        // Compose each tire's barrier function to a single smooth barrier function
        return smoothMin(smoothMin(hfl, hfr), smoothMin(hrl, hrr));
    }

    // Compute the gradient of the barrier function using finite-differences 
    VectorXf dPhi(VectorXf x)
    {
        float epsilon = 1e-3;
        int n = x.rows();
        VectorXf d_phi(n);
        MatrixXf eye = epsilon*MatrixXf::Identity(n, n);
        for (int i = 0; i < n; i++)
        {
            d_phi(i) = (phi(x + eye.col(i)) - phi(x - eye.col(i)))/2/epsilon;
        }
        return d_phi;
    }

    // Compute the lie-derivative of the barrier function projected onto the manifold of fs(x)
    float LfPhi(VectorXf x)
    {
        return vd.fs(x).dot(dPhi(x));
    }

    // Compute the lie-derivative of the barrier function projected onto the manifold of gs(x)
    MatrixXf LgPhi(VectorXf x)
    {
        return dPhi(x).transpose()*vd.gs(x);
    }

    // Compute a single discrete-step of the closed-loop dynamics
    VectorXf stepODE(float t, VectorXf x)
    {
        int n = x.rows();
        MatrixXf K = MatrixXf::Zero(n, 4);
        K.col(0) = vd.func(t, x);
        K.col(1) = vd.func(t + Ts/2, x + K.col(0)/2);
        K.col(2) = vd.func(t + Ts/2, x + K.col(1)/2);
        K.col(3) = vd.func(t + Ts, x + K.col(2));
        x += Ts*(K.col(0) + 2*K.col(1) + 2*K.col(2) + K.col(3))/6 + Ts*randNormal(n, 2.5);
        return x;

    }

    // Calculate the probability of system safety
    float F(float t, VectorXf x)
    {
        int k = round(t/Ts);
        VectorXf safeProb = VectorXf::Zero(N_episode);
        VectorXf x_s;
        float p;
        float t_step;
        for (int i = 0; i < N_episode; i++)
        {
            x_s = x;
            p = 1.0;
            t_step = t;
            for (int j = 0; j < N_horizon; j++)
            {
                x_s = stepODE(t_step, x_s);
                t_step += Ts;
                if (phi(x_s) <= 0.00)
                {
                    p = 0.0;
                    break;
                }
            }
            safeProb(i, 0) = p;
        }
        return safeProb.mean();
    }

    // Solve the QP and obtain the safety-aware control action
    VectorXf safeCtrl(VectorXf x, VectorXf uN)
    {
        int m = uN.rows();
        MatrixXf Gi = LgPhi(x);
        VectorXf hi(1); hi << LfPhi(x) + 1.00*phi(x);
        VectorXf u = solveQP(uN, Gi, hi);
        return u;
    }
};

// Implementation of generalized RK4 ODE solver
class SolveODE
{
    public:

    float Ts; // Discrete time step
    int n, N; // State vector dimension, Number of steps
    VectorXf x0; // Initial state vector

    function<VectorXf(float, VectorXf)> func; // System dynamics function x_dot = func(t, x)

    SolveODE(float Ts_, int N_, VectorXf x0_, function<VectorXf(float, VectorXf)> func_)
    {
        Ts = Ts_;
        N = N_;
        x0 = x0_;
        func = func_;
        n = x0.rows();
    }

    MatrixXf solve()
    {
        MatrixXf X = MatrixXf::Zero(n, N); // Matrix to store solved ODE
        MatrixXf K = MatrixXf::Zero(n, 4); // Matrix to store intermediate values 

        X.col(0) = x0; // Initialize state

        float t = 0; // Initialize time

        for (int k = 0; k < N-1; k++)
        {
            t = k*Ts; // Compute current time-step

            // Run RK4 algorithm
            K.col(0) = func(t, X.col(k));
            K.col(1) = func(t + Ts/2, X.col(k) + K.col(0)/2);
            K.col(2) = func(t + Ts/2, X.col(k) + K.col(1)/2);
            K.col(3) = func(t + Ts, X.col(k) + K.col(2));

            X.col(k+1) = X.col(k) + Ts*(K.col(0) + 2*K.col(1) + 2*K.col(2) + K.col(3))/6;
        }
        
        return X; // Return solved ODE
    }
};

int main()
{
    cout << "Solving ODE... " << endl;
    auto start_time = chrono::system_clock::now(); // Note time of start of execution


    float v0 = 8.00;
    float R = 0.316;
    float Ts = 1e-3;
    float a_factor = 1.00;
    int N = 25000-80;
    int N_horizon = 1000;
    int N_episodes = 100;

    CSVData rd("traj_data.csv");
    traj_data = rd.readFromCSVfile(); // Load planned state and control trajectories

    ArrayXf x0(11);
    x0 << traj_data(0, 0), traj_data(1, 0),  traj_data(2, 0),  1.41964e+01,
       -1.38715e-03,  1.30928e-01,  4.45212e+01,  4.51403e+01,
        4.45322e+01,  4.51511e+01,  2.61690e-02; // Initial state based on planned trajectory and analytical calculations

    StochasticCBF scbf(Ts, N_horizon, N_episodes);

    VehicleDynamics dynamics;

    function<VectorXf(VectorXf, VectorXf)> f_sys = [&](VectorXf x, VectorXf u) { return dynamics.f_sys(x, u); }; // Cast the open-loop vehicle dynamics function onto a lambda expression

    function<VectorXf(float t, VectorXf)> safe_ctrl = [&](float t, VectorXf x) { return scbf.safeCtrl(x, dynamics.ctrl(t, x)); }; // Nest the MPC controller into the safe controller and cast it onto a lambda expression

    function<VectorXf(float t, VectorXf)> func = [&](float t, VectorXf x) { return f_sys(x, safe_ctrl(t, x)); }; // Return the complete closed-loop autonomous system dynamics that; ready to be solved by RK4 ODE solver

    SolveODE ode(Ts, N, x0, func); // Create ODE solver instance; here we pass the complete closed-loop autonomous system dynamics 

    MatrixXf X_sol = ode.solve(); // Solve ODE and obtain solution

    auto stop_time = chrono::system_clock::now();
    chrono::duration<double> diff = stop_time - start_time; // Stop the timer and record the solver time

    CSVData sv("X_data.csv", X_sol);

    sv.writeToCSVfile(); // Write the ODE solution to CSV file

    cout << "Solve Complete! And time to run was: " << diff.count() << endl; // Print the ODE solver time

    return 0;
}
