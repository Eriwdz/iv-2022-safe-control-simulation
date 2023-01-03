/*
Created on Feb 08 11:06:07 2022
​
@author: siddg

This header file implements a sparse linear time varying model predictive controller for non-holonomic systems.
Currently, state constrains are omitted from the optimization problem since they are accounted for while planning.
*/


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct>

// Declare namespaces

using namespace Eigen;
using namespace std;

// Define commonly used datatypes

// SparseMatrix datatype
typedef SparseMatrix<float> SpMat;

// Sparse LU Factorization Natural Ordering Permutation Matrix (Identity matrix)
typedef NaturalOrdering<int> Natural;

// Sparse LU Factorization Colamd Ordering Permutation Matrix (To prevent fill-ins during matrix factorization needed for maintaining sparsity)
typedef COLAMDOrdering<int> COLAMD;

// Main MPC Class
class MPC
{
    public:
    // Define the optimization penalty matrices Q, R; Linearized Dynamics matrices A, B; LMI matrices G, H for QP solver
    SpMat Q, R, A, B, G, H;

    // Optimization Vectors
    VectorXf c, g, h;

    // State dimensions - n; control dimensions m; MPC preview horizon N
    int n, m, N;

    // Linearization Step-size - delta; vehicle base-length (front to rear mid-axle distance) - L; discrete time-step - Ts
    float delta, L, Ts;
    
    SparseLU <SpMat, Natural> solver; // Initialize Sparse-LU solver

    // Class constructor - call in initial planned state, planned control, desired penalty matrices, and MPC horizon
    MPC(VectorXf x0, VectorXf u0, MatrixXf Q__, MatrixXf R__, float L_, float Ts_, int N_)
    {
        delta = 1e-6;
        L = L_;
        Ts = Ts_;
        N = N_;
        n = Q__.cols();
        m = R__.cols();
        MatrixXf A_, B_, AB_;

        // Get linearized dynamics matrices A and B 
        AB_ = jacobian(x0, u0); 
        A_ = AB_(seq(0, n-1), seq(0, n-1));
        B_ = AB_(all, seq(n, n+m-1));

        // Typecast dense matrices to sparse matrices
        SpMat Q_ = Q__.sparseView();
        SpMat R_ = R__.sparseView();

        // Use Kronecker Product to "lift" the optimization matrices to fit the MPC horizon
        Q = KroneckerProductSparse(MatrixXf::Identity(N, N).sparseView(), Q_);
        R = KroneckerProductSparse(MatrixXf::Identity(N, N).sparseView(), R_);

        // Create the initial lifted linear system dynamics
        MatrixXf A_block = MatrixXf::Zero(n*N, n*N);
        MatrixXf B_block = MatrixXf::Zero(n*N, m*N);
        for (int j = 0; j < N-1; j++)
        {
            A_block.block((j+1)*n, j*n, n, n) = -A_;
            B_block.block(j*n, j*m, n, m) = B_;
        }
        B_block.block((N-1)*n, (N-1)*m, n, m) = B_;
        A = KroneckerProductSparse(MatrixXf::Identity(N, N).sparseView(), MatrixXf::Identity(n, n).sparseView()) + A_block.sparseView();
        B = B_block.sparseView();
        
        Q.block(n*(N-1), n*(N-1), n, n) *= 10;
        c = VectorXf::Zero(n*N);

    }
    // Compute jacobian of system dynamics
    MatrixXf jacobian(VectorXf x, VectorXf u)
    {
        int n = x.rows();
        int m = u.rows();
        MatrixXf A(n, n);
        MatrixXf B(n, m);
        MatrixXf AB(n, n+m);

        A << 1, 0, -Ts*u(0)*sin(x(2)),
             0, 1, Ts*u(0)*cos(x(2)),
             0, 0, 1;

        B << Ts*cos(x(2)), 0,s
             Ts*sin(x(2)), 0,
             Ts/L*tan(u(1)), Ts*u(0)/cos(u(1))/cos(u(1));

        AB << A, B;

        return AB;
        
    }
    // Compute optimal control action for given current-state, planned state-trajectory and planned control-trajectory
    VectorXf solveMPC(VectorXf x, MatrixXf X_ref, MatrixXf U_ref)
    {
        MatrixXf A_block = MatrixXf::Zero(n*N, n*N);
        MatrixXf B_block = MatrixXf::Zero(n*N, m*N);
        MatrixXf AB_;

        // Compute linearized dynamics for planned state and control actions; lifted to meet the required MPC horizon
        AB_ = jacobian(X_ref.col(0), U_ref.col(0));
        c.segment(0, n) = AB_(seq(0, n-1), seq(0, n-1))*(x - X_ref.col(0));
        for (int j = 0; j < N-1; j++)
        {   
            AB_ = jacobian(X_ref.col(j), U_ref.col(j));
            A_block.block((j+1)*n, j*n, n, n) = -AB_(seq(0, n-1), seq(0, n-1));
            B_block.block(j*n, j*m, n, m) = AB_(all, seq(n, n+m-1));
        }
        AB_ = jacobian(X_ref.col(N-1), U_ref.col(N-1));
        B_block.block((N-1)*n, (N-1)*m, n, m) = AB_(all, seq(n, n+m-1));
        A = KroneckerProductSparse(MatrixXf::Identity(N, N).sparseView(), MatrixXf::Identity(n, n).sparseView()) + A_block.sparseView();
        B = B_block.sparseView();

        // Solve the optimization problem to obtain optimal control action
        solver.compute(A);
        g = solver.solve(c);
        solver.compute(A);
        G = solver.solve(B);
        H = R + G.transpose()*Q*G;
        h = G.transpose()*Q*g;
        solver.compute(H);
        Map<VectorXf> U_flat(U_ref.data(), U_ref.size());
        VectorXf U = solver.solve(-h) + U_flat;
        return U;
    }
};