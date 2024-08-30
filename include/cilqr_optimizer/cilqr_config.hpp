#pragma once

#include <vector>

class CilqrConfig
{
public:
    // iLQR iteration parameters
    struct Ilqr
    {
        double update_freq = 0.1;           // Planning frequency
        double dense_timestep = 0.1;        // Time step in short future
        double sparse_timestep = 0.1;       // Time step in far future
        int horizon = 50;                   // Total horizon (number of control input)
        int dense_horizon = 50;             // Size of far future
        int feasibility_check_horizon = 50; // Only part of the planning horizon
                                            // is applied with feasibility check

        double tol = 0.00008;  // CiLQR convergence tolerance
        int max_iter = 50;     // Maximum CiLQR iterations
        int min_iter = 3;      // Minimum CiLQR iterations
        int num_states = 6;    // Number of state variables
        int num_ctrls = 2;     // Number of control variables
        int num_auxiliary = 2; // Number of auxiliary terms

        double lambda_factor = 1.618; // Normalization factor
        double max_lambda = 1000.0;   // Maximum normalization factor
        double wheel_base = 2.85;     // TODO What is this
    } ilqr;

    // Cost weight and barrier cost param
    struct CostWeight
    {
        std::vector<double> w_states = {1.0, 16.0, 0.0, 0.0, 0.0, 0.0}; // state weight for [x,y,vel,theta,acc,yawrate]
        std::vector<double> w_controls = {1.0, 10.0};                // control weight for [jerk, yawrate_dot]
        std::vector<double> w_auxiliary = {2.0, 1.0};                 // auxiliary weight for [lat_acc and lat_jerk]
        // std::vector<double> w_states_regulation;   // state regularization weight
        // std::vector<double> w_controls_regulation; // control regularization weight

        std::vector<double> q1_states = {0.0, 0.0, 3.0, 0.1, 0.1, 0.1}; // barrier cost param q1 for state related constraints for [x,y,vel,theta,acc,yawrate]
        std::vector<double> q2_states = {0.0, 0.0, 5.0, 3.0, 3.0, 3.0}; // barrier cost param q2 for state related constraints for [x,y,vel,theta,acc,yawrate]

        std::vector<double> q1_controls = {1.0, 3.0}; // barrier cost param q1 for control related constraints for [jerk, yawrate_dot]
        std::vector<double> q2_controls = {1.0, 3.0}; // barrier cost param q2 for control related constraints for [jerk, yawrate_dot]

        double q1_boundary = 1.0; // boundary constraints barrier cost param q1
        double q2_boundary = 3.0; // boundary constraints barrier cost param q2

        double q1_front = 5.0; // front collision circle boundary constraints param
        double q2_front = 3.0; // front collision circle boundary constraints param

        double q1_rear = 5.0; // rear collision circle boundary constraints param
        double q2_rear = 3.0; // rear collision circle boundary constraints param
    } cost_weight;

    // Constraints upper bound and lower bound,
    // as well as collision buffer
    struct Constraint
    {
        // Safe distance to boundary
        double boundary_safe = 1.0;

        // Safe distance to boundary when applying feasibility checks
        double feasibility_check_boundary_safe = 0.5;

        // State constraints lower bound
        std::vector<double> state_constraints_min = {0.0, 0.0, 0.0, -3.14, -5.0, -0.2}; // state_constraints_min for [x,y,vel,theta,acc,yawrate]

        // State constraints upper bound
        std::vector<double> state_constraints_max = {0.0, 0.0, 30.0, 3.14, 2.0, 0.2}; // state_constraints_max for [x,y,vel,theta,acc,yawrate]

        // Control constraints lower bound
        std::vector<double> control_constraints_min = {-10.0, -0.3}; // control_constraints_min for  [jerk, yawrate_dot]

        // Control constraints upper bound
        std::vector<double> control_constraints_max = {10.0, 0.3}; // control_constraints_max for [jerk, yawrate_dot]

        // State and control boundaries when applying feasibility check
        std::vector<double> feasibility_check_state_constraints_min = {0.0, 0.0, 0.0, -3.14, -5.0, -0.2};
        std::vector<double> feasibility_check_state_constraints_max = {0.0, 0.0, 30.0, 3.14, 4.0, 0.2};
        std::vector<double> feasibility_check_control_constraints_min = {-10.0, -0.3};
        std::vector<double> feasibility_check_control_constraints_max = {10.0, 0.3};

        // Specific boundary constraints
        std::vector<double> max_vel_constraints;
        std::vector<double> min_vel_constraints;
        std::vector<double> max_yaw_rate_constraints;
        std::vector<double> min_yaw_rate_constraints;
        std::vector<double> max_acc_constraints;
        std::vector<double> min_acc_constraints;

        // Dynamic obstacle safety buffer
        struct Obstacle
        {
            double t_safe = 0.4; // time buffer
            double a_safe = 1.0; // major axis buffer
            double b_safe = 0.4; // minor axis buffer
            double r_safe = 1.5; // coverage circle radius with buffer
        } obstacle;
    } constraint;

    // Vehicle size info
    struct EgoVehicle
    {
        double wheel_base = 2.85; // [m]
        double width = 1.86;             // [m]
        double length = 4.93;            // [m]
    } ego_vehicle;
};