/*
 * Copyright (C) 2022 by SenseTime Group Limited. All rights reserved.
 *
 */
#pragma once

#include <iostream>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "eigen3/Eigen/Core"
#include "osqp/osqp.h"

namespace senseAD {
namespace perception {
namespace speed {
// #ifdef max
// #undef max
// #endif  // max
// #ifdef min
// #undef min
// #endif  // min

struct IDMParam {
    double desire_velocity{16.7};
    double min_spacing{10.0};
    double time_headway{1.0};
    double max_acc{2.0};
    double desired_deacc{1.2};
    double min_dacc{3.5};
    double max_jerk{6.0};
    double delta{5.0};
};

class IDMPlanner {
 public:
    /**
     * @brief CalcAcc calculate acceleration by IDM formula
     *
     * @param delta_s is distance to the leading vehicle
     * @param curr_v is current velocity
     * @param delta_v is velocity difference to the leading vehilce,
     * i.e. delta_v = curr_v - v_leading
     *
     * @return acc based on idm, which will clipped by min_acc and max_jerk
     */
    static double CalcIDMAcc(const IDMParam &idm_param,
                             const double delta_s,
                             const double curr_v,
                             const double delta_v) {
        // desired s should not be less than min_spacing
        double s_desire =
            idm_param.min_spacing +
            std::max(0.0, curr_v * idm_param.time_headway +
                              curr_v * delta_v /
                                  std::sqrt(idm_param.desired_deacc *
                                            idm_param.max_acc) /
                                  2.0);
        double acc_idm =
            idm_param.max_acc *
            (1 - std::pow(curr_v / idm_param.desire_velocity, idm_param.delta) -
             std::pow(s_desire / delta_s, 2));
        return std::max(-idm_param.min_dacc, acc_idm);
    }

    /**
     * @brief CalcFreeAcc calculate acceleration for vehicle in free status
     *
     * @param curr_v is current velocity
     *
     * @return acceleration
     */
    static double CalcFreeAcc(const IDMParam &idm_param, const double curr_v) {
        double idm_acc = idm_param.max_acc *
                         (1.0 - std::pow(curr_v / idm_param.desire_velocity,
                                         idm_param.delta));
        return std::max(-1.0 * idm_param.min_dacc,
                        std::min(idm_acc, idm_param.max_acc));
    }

    /**
     * @brief ChooseLeader choose leader by comparing acceleration
     *
     * @param idm_param is the idm param
     * @param curr_v is the current velocity
     * @param delta_s_1 is the dist to first leader
     * @param delta_v_1 is the delta speed to first leader
     * @param delta_s_2 is the dist to second leader
     * @param delta_v_2 is the delta speed to second leader
     *
     * @return true if first leader should be picked, otherwise pick the second
     */
    static bool ChooseLeader(const IDMParam &idm_param,
                             const double curr_v,
                             const double delta_s_1,
                             const double delta_v_1,
                             const double delta_s_2,
                             const double delta_v_2) {
        return CalcIDMAcc(idm_param, delta_s_1, curr_v, delta_v_1) <
               CalcIDMAcc(idm_param, delta_s_2, curr_v, delta_v_2);
    }
};

struct SpeedOptimizeWeight {
    double s_weight{70.0};
    double vel_weight{10.0};
    double acc_weight{10.0};
    double jerk_weight{50.0};
    double curve_weight{0};
};
struct OptimizeHorizon {
    int state_dim{3};
    int control_dim{1};
    int horizon_interval{5};
    int horizon_step{40};
    int cost_function_state_dim{9};
    int cost_function_control_dim{1};
    int constraint_state_dim{3};
    int constraint_control_dim{2};
    double resolution{0.2};
};
struct SpeedOptimizeControl {
    SpeedOptimizeControl() = default;
    explicit SpeedOptimizeControl(const double jerk_input) : jerk(jerk_input) {}
    double jerk = 0.0;
};
struct SpeedOptimizeState {
    SpeedOptimizeState() = default;
    SpeedOptimizeState(const double s_input,
                       const double vel_input,
                       const double acc_input)
        : s_frenet(s_input), vel(vel_input), acc(acc_input) {}
    double s_frenet{0.0};
    double vel{0.0};
    double acc{0.0};
};
struct SpeedOptimizeProfile {
    double t_seq = 0.0;
    double s_ref = 0.0;
    double v_ref = 0.0;
    double s_upper = 0.0;
    double s_lower = 0.0;
    double curvature = 0.0;
    double a_upper = 0.0;
    double a_lower = 0.0;
    double v_upper = 0.0;
    double v_lower = 0.0;
    double confidence_factor = 0.0;
    double sr_factor = 0.0;
    double s_leader = 0.0;
    double v_leader = 0.0;
    double static_follow_distance = 0.0;
    double ttc = 0.0;
    double jerk_lower = 0.0;
    double jerk_upper = 0.0;
    double s_upper_enable = 0.0;
    double s_follower = 0.0;
    double s_ref_cruise = 0.0;
    double v_ref_cruise = 0.0;
    double a_ref_cruise = 0.0;
    double s_ref_collision = 0.0;
    double v_ref_collision = 0.0;
    double a_ref_collision = 0.0;
    double jerk_ref = 0.0;
    double collision_probability = 0.0;
    double collision_penalty_factor = 0.0;
    double jerk_ref_factor = 0.0;
    int exist_leader = 0;
    std::string leader_id = "";
};
inline double Relu(double x, double relu_decay) {
    return (std::log(std::exp(relu_decay * x) + 1) / relu_decay);
}

// linearized RELU funciton at x_0
// (log(exp(relu_decay * (x-x_r)) + 1) / relu_decay)
// (log(exp(relu_decay * (x_0-x_r)) + 1) / relu_decay) + exp(relu_decay *
// (x_0-x_r)) * (x - x_0) / (exp(relu_decay * (x_0-x_r)) + 1)
inline double ReluDiff(double x, double relu_decay) {
    return std::exp(relu_decay * x) / (std::exp(relu_decay * x) + 1);
}

namespace OSQPHint {
// default,follow,redlight,emergency
const std::map<std::string, std::vector<float>> weight_config = {
    {"default", {5.0, 5.0, 10.0, 100.0, 0}},
    {"follow", {70.0, 10.0, 10.0, 50, 0}},
    {"redlight", {4.3, 200.0, 1.0, 300.0, 0}},
    {"emergency", {25.0, 2.0, 10.0, 10.0, 0}}};
const std::unordered_map<int, std::string> StateErrorTab = {
    {4, "OSQP_DUAL_INFEASIBLE_INACCURATE"},
    {3, "OSQP_PRIMAL_INFEASIBLE_INACCURATE"},
    {2, "OSQP_SOLVED_INACCURATE"},
    {1, "OSQP_SOLVED"},
    {-2, "OSQP_MAX_ITER_REACHED"},
    {-3, "OSQP_PRIMAL_INFEASIBLE"},
    {-4, "OSQP_DUAL_INFEASIBLE"},
    {-5, "OSQP_SIGINT"},
    {-6, "OSQP_TIME_LIMIT_REACHED"},
    {-7, "OSQP_NON_CVX"},
    {-10, "OSQP_UNSOLVED"}};

const std::unordered_map<int, std::string> SetupErrorTab = {
    {1, "OSQP_DATA_VALIDATION_ERROR"},
    {2, "OSQP_SETTINGS_VALIDATION_ERROR"},
    {3, "OSQP_LINSYS_SOLVER_LOAD_ERROR"},
    {4, "OSQP_LINSYS_SOLVER_INIT_ERROR"},
    {5, "OSQP_NONCVX_ERROR"},
    {6, "OSQP_MEM_ALLOC_ERROR"},
    {7, "OSQP_WORKSPACE_NOT_INIT_ERROR"}};
}  // namespace OSQPHint

// solve the mpc problem in the form of
// minimize:
// 0.5*sigma_0^n{(cf_x*x_k-x_r_k)'*q_k*(cf_x*x_k-x_r_k)} +
// 0.5*sigma_0^n-1{(u_k-u_r_k)'*r_k*(u_k-u_r_k)}
// subject to: Inequlity Constraints:
// x_lower_k <= c_x_k*x_k <= x_upper_k, u_lower_k <= c_u_k*u <= u_upper_k
// Equlity Constraints: x_dot = a*x + b*u + c, x_0 = x_init

// transfer to osqp form
// minimize: 0.5*X'*P*X + Q'*X
// subject to: L <= A*X <= U
class MpcOsqp {
 public:
    using Ptr = std::shared_ptr<MpcOsqp>;
    MpcOsqp();
    virtual ~MpcOsqp();

    enum class Status {
        SUCCESS = 0,
        ERROR_INPUT = -1,     // input error, for example, the matrix dimension
                              // does not match
        ERROR_NULL_PTR = -2,  // null ptr occurs when setup
        ERROR_SETUP = -3,     // error when setup osqp
        ERROR_OUTPUT_NAN = -4,               // output solution has nan value
        ERROR_OSQP = 1,                      // osqp error
        ERROR_OSQP_STATUS = ERROR_OSQP + 1,  // osqp return status error
        ERROR_OSQP_SOLUTION =
            ERROR_OSQP + 2,  // osqp return solution is nullptr

        ERROR_MODEL = 10,  // nn model error
    };

    /**
     * @brief Input of the osqp mpc solver
     * @param ad The system dynamic matrix of linear system
     * @param bd The control matrix of linear system
     * @param cd Residual matrix after linearization
     * @param coeff_A the coefficient matrix of high order linear system
     * @param q The cost matrix for state
     * @param r The cost matrix for control
     * @param x_initial The initial state matrix
     * @param cf_x State to state cost function matrix
     * @param x_ref Reference of state
     * @param u_ref Reference of control
     * @param c_x Constraint matrix of state
     * @param c_u Constraint matrix of control
     * @param u_lower The lower bound control constrain matrix
     * @param u_upper The upper bound control constrain matrix
     * @param x_lower The lower bound state constrain matrix
     * @param x_upper The upper bound state constrain matrix
     * @param interval Interval of discrete system
     * @param horizon The prediction horizon
     * @param state_dim State dimension
     * @param control_dim Control dimension
     * @param state_constraint_dim State constraint matrix dimension
     * @param control_constraint_dim Control constraint matrix dimension
     * @param state_cost_function_dim State cost function matrix dimension
     * @param max_iter The maximum iterations
     * @param eps_abs Absolute convergence tolerance
     * @param eps_rel Relative convergence tolerance
     * @param eps_prim_inf Prim inf tolerance, for primal infeasibility check
     * @param discretize_order Order of discretized accuracy
     */
    struct Input {
        // online data
        Eigen::MatrixXd x_initial{};
        std::vector<Eigen::MatrixXd> ad = {};
        std::vector<Eigen::MatrixXd> bd = {};
        std::vector<Eigen::MatrixXd> cd = {};
        std::vector<Eigen::MatrixXd> coeff_A = {};
        std::vector<Eigen::MatrixXd> q = {};
        std::vector<Eigen::MatrixXd> r = {};
        std::vector<Eigen::MatrixXd> cf_x = {};
        std::vector<Eigen::MatrixXd> x_ref = {};
        std::vector<Eigen::MatrixXd> u_ref = {};
        std::vector<Eigen::MatrixXd> c_x = {};
        std::vector<Eigen::MatrixXd> c_u = {};
        std::vector<Eigen::MatrixXd> u_lower = {};
        std::vector<Eigen::MatrixXd> u_upper = {};
        std::vector<Eigen::MatrixXd> x_lower = {};
        std::vector<Eigen::MatrixXd> x_upper = {};
        // dimensions
        size_t state_dim = 0;
        size_t control_dim = 0;
        size_t state_constraint_dim = 0;
        size_t control_constraint_dim = 0;
        size_t state_cost_function_dim = 0;
        int horizon;
        // settings
        int max_iter = 0;
        double eps_abs = 0.0;
        double eps_rel = 0.0;
        double eps_prim_inf = 0.0;
        int discretize_order = 0;
    };

    /**
     * @brief execution of the solver
     * @param output output of the solution
     * @param input online data
     * @param debug_print is debug print
     */
    Status Solve(const Input &input,
                 const bool debug_print,
                 std::vector<double> *output);

 private:
    bool CheckInputOK(const Input &input);

    bool CheckMatrixRowsCols(const std::vector<Eigen::MatrixXd> &matrix,
                             size_t vec_size,
                             size_t matrix_rows,
                             size_t matrix_cols);

    // TODO(zhangsichao): return status
    void CalculateCostFunction(std::vector<c_float> *P_data,
                               std::vector<c_int> *P_indices,
                               std::vector<c_int> *P_indptr);

    void CalculateConstraintMatrix(std::vector<c_float> *A_data,
                                   std::vector<c_int> *A_indices,
                                   std::vector<c_int> *A_indptr);

    void CalculateConstraintBound();

    void Settings();

    void Data();

    void Update();

    Status CheckSolveStatus();

    void InitializeMatrix();

    Status Init();

    void FreeData(OSQPData *data) {
        c_free(data->A);
        c_free(data->P);
        c_free(data);
    }

    template <typename T>
    T *CopyData(const std::vector<T> &vec) {
        T *data = new T[vec.size()];
        memcpy(data, vec.data(), sizeof(T) * vec.size());
        return data;
    }

 private:
    bool debug_print_ = false;
    // osqp problem form
    Input mpc_{};
    size_t state_cost_function_dim_;
    size_t osqp_state_dim_;       // dim of state matrix x is osqp form
    size_t osqp_constraint_dim_;  // row of constraint matrix A in osqp form
    Eigen::VectorXd gradient_;
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
    std::vector<c_float> P_data_ = {};
    std::vector<c_int> P_indices_ = {};
    std::vector<c_int> P_indptr_ = {};
    std::vector<c_float> A_data_ = {};
    std::vector<c_int> A_indices_ = {};
    std::vector<c_int> A_indptr_ = {};

    OSQPWorkspace *osqp_workspace_{nullptr};
    OSQPSettings *osqp_settings_{nullptr};
    OSQPData *osqp_data_{nullptr};

    bool is_last_solved_ = false;  // is last solution success
    bool is_setup_ = false;        // is osqp solver setup
};
}  // namespace speed
}  // namespace perception
}  // namespace senseAD
