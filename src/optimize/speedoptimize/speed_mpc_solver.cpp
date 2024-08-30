/*
 * Copyright (C) 2023 by SenseTime Group Limited. All rights reserved.
 *
 */

#include <utility>
#include "speed_mpc_solver.hpp"

namespace senseAD {
namespace perception {
using namespace senseAD::perception::speed;
namespace camera {
SpeedMpcSolver::SpeedMpcSolver(const OptimizeHorizon& horizon)
    : state_dim_(horizon.state_dim),
      control_dim_(horizon.control_dim),
      state_constraint_dim_(horizon.constraint_state_dim),
      state_cost_function_dim_(horizon.cost_function_state_dim),
      control_constraint_dim_(horizon.constraint_control_dim),
      horizon_step_(horizon.horizon_step) {
    solver_type_ = "speed";
}

void SpeedMpcSolver::Update(const SpeedOptimizeState& initial_state,
                            const std::vector<SpeedOptimizeProfile>& profile,
                            const SpeedOptimizeWeight& weight) {
    profile_ = profile;
    initial_state_ = initial_state;
    weight_ = weight;
}

void SpeedMpcSolver::GetCostFunction(MpcOsqp::Input* input) const {
    constexpr double unsafe_penalty_rate = 2.0;
    const double unsafe_cost = unsafe_penalty_rate * weight_.s_weight;
    constexpr double surpass_cost_weight = 0.2;
    constexpr double surpass_max_cost_distance = 10.0;
    constexpr double surpass_min_cost_distance = -10.0;
    constexpr double safe_bound = 5.0;
    constexpr double unsafe_cost_ratio = 10.0;
    constexpr double relu_decay = 1.0;

    input->q.assign(horizon_step_ + 1,
                    Eigen::MatrixXd::Zero(state_cost_function_dim_,
                                          state_cost_function_dim_));
    input->r.assign(horizon_step_,
                    Eigen::MatrixXd::Zero(control_dim_, control_dim_));

    for (size_t i = 0; i < horizon_step_ + 1; ++i) {
        double collision_factor = profile_[i].collision_penalty_factor;
        double cruise_factor = 1.0 - collision_factor;
        input->q[i](0, 0) = weight_.s_weight * cruise_factor *
                            profile_[i].confidence_factor *
                            profile_[i].exist_leader * profile_[i].sr_factor *
                            profile_[i].confidence_factor *
                            profile_[i].exist_leader * profile_[i].sr_factor;
        input->q[i](1, 1) = weight_.vel_weight * cruise_factor *
                            profile_[i].confidence_factor *
                            profile_[i].confidence_factor;
        input->q[i](2, 2) = weight_.acc_weight * cruise_factor;
        input->q[i](6, 6) =
            unsafe_cost * profile_[i].confidence_factor *
            profile_[i].exist_leader * (1 - profile_[i].s_upper_enable) *
            unsafe_cost_ratio * profile_[i].confidence_factor *
            profile_[i].exist_leader * (1 - profile_[i].s_upper_enable) *
            unsafe_cost_ratio;
        input->q[i](7, 7) = weight_.curve_weight * profile_[i].curvature *
                            profile_[i].curvature;

        if (i < horizon_step_) {
            input->q[i](3, 3) =
                weight_.s_weight * collision_factor *
                profile_[i].confidence_factor * profile_[i].exist_leader *
                profile_[i].sr_factor * profile_[i].confidence_factor *
                profile_[i].exist_leader * profile_[i].sr_factor;
            input->q[i](4, 4) = weight_.vel_weight * collision_factor *
                                profile_[i].confidence_factor *
                                profile_[i].confidence_factor;
            input->q[i](5, 5) = weight_.acc_weight * collision_factor;
            input->q[i](8, 8) = surpass_cost_weight * surpass_cost_weight;
        }
    }
    for (size_t i = 0; i < horizon_step_; ++i) {
        input->r[i](0, 0) = weight_.jerk_weight;
    }

    input->cf_x.assign(
        horizon_step_ + 1,
        Eigen::MatrixXd::Zero(state_cost_function_dim_, state_dim_));
    input->x_ref.assign(horizon_step_ + 1,
                        Eigen::MatrixXd::Zero(state_cost_function_dim_, 1));
    input->u_ref.assign(horizon_step_, Eigen::MatrixXd::Zero(control_dim_, 1));
    for (size_t i = 0; i < horizon_step_ + 1; ++i) {
        double s_leader_ref = profile_[i].s_leader - safe_bound;
        double s_follower_ref =
            profile_[i].s_follower - surpass_min_cost_distance;
        double range = surpass_max_cost_distance - surpass_min_cost_distance;
        input->x_ref[i](0, 0) = profile_[i].s_ref_cruise;
        input->x_ref[i](1, 0) = profile_[i].v_ref_cruise;
        input->x_ref[i](2, 0) = profile_[i].a_ref_cruise;
        input->x_ref[i](3, 0) = profile_[i].s_ref_collision;
        input->x_ref[i](4, 0) = profile_[i].v_ref_collision;
        input->x_ref[i](5, 0) = profile_[i].a_ref_collision;

        input->x_ref[i](6, 0) =
            ReluDiff(initial_state_.s_frenet - s_leader_ref, relu_decay) *
                initial_state_.s_frenet -
            Relu(initial_state_.s_frenet - s_leader_ref, relu_decay);
        input->x_ref[i](7, 0) = initial_state_.vel * initial_state_.vel;
        if (initial_state_.s_frenet - s_follower_ref > range) {
            input->x_ref[i](8, 0) = 0;
        } else {
            input->x_ref[i](8, 0) =
                ReluDiff(initial_state_.s_frenet - s_follower_ref, relu_decay) *
                    initial_state_.s_frenet -
                Relu(initial_state_.s_frenet - s_follower_ref, relu_decay);
        }

        input->cf_x[i].block(0, 0, state_dim_, state_dim_) =
            Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        input->cf_x[i].block(state_dim_, 0, state_dim_, state_dim_) =
            Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        input->cf_x[i](6, 0) =
            ReluDiff(initial_state_.s_frenet - s_leader_ref, relu_decay);
        input->cf_x[i](7, 1) = 2 * initial_state_.vel;
        if (initial_state_.s_frenet - s_follower_ref > range) {
            input->cf_x[i](8, 0) = 0;
        } else {
            input->cf_x[i](8, 0) =
                ReluDiff(initial_state_.s_frenet - s_follower_ref, relu_decay);
        }
    }
    for (size_t i = 0; i < horizon_step_; ++i) {
        input->u_ref[i](0, 0) = profile_[i].jerk_ref;
    }

    return;
}

void SpeedMpcSolver::GetConstraint(MpcOsqp::Input* input) const {
    input->c_x.assign(horizon_step_ + 1,
                      Eigen::MatrixXd::Zero(state_constraint_dim_, state_dim_));
    input->c_u.assign(
        horizon_step_,
        Eigen::MatrixXd::Zero(control_constraint_dim_, control_dim_));
    for (size_t i = 0; i < horizon_step_ + 1; ++i) {
        input->c_x[i] =
            Eigen::MatrixXd::Identity(state_constraint_dim_, state_dim_);
    }
    for (size_t i = 0; i < horizon_step_; ++i) {
        input->c_u[i] =
            Eigen::MatrixXd::Ones(control_constraint_dim_, control_dim_);
    }

    input->x_lower.assign(horizon_step_ + 1,
                          Eigen::MatrixXd::Zero(state_constraint_dim_, 1));
    input->x_upper.assign(horizon_step_ + 1,
                          Eigen::MatrixXd::Zero(state_constraint_dim_, 1));
    input->u_lower.assign(horizon_step_,
                          Eigen::MatrixXd::Zero(control_constraint_dim_, 1));
    input->u_upper.assign(horizon_step_,
                          Eigen::MatrixXd::Zero(control_constraint_dim_, 1));
    for (size_t i = 0; i < horizon_step_ + 1; ++i) {
        input->x_lower[i](0, 0) = profile_[i].s_lower;
        input->x_upper[i](0, 0) = profile_[i].s_upper;

        input->x_lower[i](1, 0) = 0.0;
        input->x_upper[i](1, 0) = profile_[i].v_upper;

        input->x_lower[i](2, 0) = profile_[i].a_lower;
        input->x_upper[i](2, 0) = profile_[i].a_upper;
    }
    for (size_t i = 0; i < horizon_step_; ++i) {
        input->u_lower[i](0, 0) = profile_[i].jerk_lower;
        input->u_upper[i](0, 0) = profile_[i].jerk_upper;

        input->u_lower[i](1, 0) = profile_[i].jerk_lower;
        input->u_upper[i](1, 0) = 6.0;
    }
    return;
}

void SpeedMpcSolver::GetLinearizedModel(MpcOsqp::Input* input) const {
    // x_initial
    input->x_initial = Eigen::MatrixXd::Zero(state_dim_, 1);
    input->x_initial(0, 0) = initial_state_.s_frenet;
    input->x_initial(1, 0) = initial_state_.vel;
    input->x_initial(2, 0) = initial_state_.acc;

    input->ad.assign(horizon_step_,
                     Eigen::MatrixXd::Zero(state_dim_, state_dim_));
    input->bd.assign(horizon_step_,
                     Eigen::MatrixXd::Zero(state_dim_, control_dim_));
    input->cd.assign(horizon_step_, Eigen::MatrixXd::Zero(state_dim_, 1));

    // x_dot = Ax + Bu + C
    // matrix_A
    std::vector<Eigen::MatrixXd> matrix_A;
    Eigen::MatrixXd matrix_a = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    matrix_a(0, 1) = 1.0;
    matrix_a(1, 2) = 1.0;
    matrix_A.assign(horizon_step_ + 1, matrix_a);

    // matrix_B
    std::vector<Eigen::MatrixXd> matrix_B;
    Eigen::MatrixXd matrix_b = Eigen::MatrixXd::Zero(state_dim_, control_dim_);
    matrix_b(2, 0) = 1.0;
    matrix_B.assign(horizon_step_ + 1, matrix_b);

    Eigen::MatrixXd state_identity_matrix =
        Eigen::MatrixXd::Identity(state_dim_, state_dim_);

    // three order discretized model
    input->discretize_order = 3;

    // calculate finite difference coefficient matrix
    Eigen::MatrixXd model_coeff =
        Eigen::MatrixXd::Zero(input->discretize_order + 1, 1);
    std::vector<double> t_seq(profile_.size(), 0.0);
    for (size_t i = 0; i < profile_.size(); ++i) {
        t_seq[i] = profile_[i].t_seq;
    }
    std::vector<Eigen::MatrixXd> coeff_A(t_seq.size(), model_coeff);
    if (!GetLinearizedModelCoeff(input->discretize_order, t_seq, &coeff_A)) {
        std::cerr << "get speed linearized coefficient failed!";
        return;
    }
    // (A - a)f(x0) + Bu + C = bf(x1) + cf(x2) + df(x3)
    for (size_t i = 0; i < horizon_step_; ++i) {
        input->ad[i] =
            matrix_A.at(i) - state_identity_matrix * coeff_A[i](0, 0);
        input->bd[i] = matrix_B.at(i);
        input->cd[i] = Eigen::MatrixXd::Zero(state_dim_, 1);
    }
    input->coeff_A = std::move(coeff_A);
    return;
}

bool SpeedMpcSolver::GetLinearizedModelCoeff(
    const int discretize_order,
    const std::vector<double>& interval_seq,
    std::vector<Eigen::MatrixXd>* matrix_coeff_vec) const {
    if (interval_seq.empty()) {
        std::cerr << "GetLinearizedModelCoeff : interval seq is empty";
        return false;
    }
    int matrix_dim = discretize_order + 1;
    // for fixed pattern, limit the scope of local variables
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(matrix_dim, matrix_dim);
        Eigen::VectorXd b = Eigen::MatrixXd::Zero(matrix_dim, 1);
        for (size_t i = 0; i < interval_seq.size() - discretize_order; ++i) {
            std::vector<double> delta_t(interval_seq.begin() + i,
                                        interval_seq.begin() + i + matrix_dim);
            for (size_t j = 1; j < delta_t.size(); ++j) {
                delta_t[j] = delta_t[j] - delta_t.front();
            }
            delta_t.front() = 0;

            int factor = 1;
            A.row(0) = Eigen::MatrixXd::Ones(1, matrix_dim);
            for (size_t j = 1; j < matrix_dim; ++j) {  // row
                factor = factor * j;
                Eigen::MatrixXd tmp_row(1, matrix_dim);
                for (size_t k = 0; k < matrix_dim; ++k) {  // col
                    tmp_row(0, k) = pow(delta_t[k], j) / factor;
                }
                A.row(j) = tmp_row;
            }

            b(1) = 1;
            if (A.determinant() == 0) {
                std::cerr << "Matrix A is not invertible";
                return false;
            }
            matrix_coeff_vec->at(i) = A.colPivHouseholderQr().solve(b);
        }
    }

    // for those last points which are less than discretize_order
    int n = interval_seq.size() - discretize_order;
    for (size_t i = 1; i < discretize_order; ++i) {
        matrix_dim = discretize_order + 1 - i;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(matrix_dim, matrix_dim);
        Eigen::VectorXd b = Eigen::MatrixXd::Zero(matrix_dim, 1);
        std::vector<double> delta_t(interval_seq.begin() + n,
                                    interval_seq.begin() + n + matrix_dim);

        for (size_t j = 1; j < delta_t.size(); ++j) {
            delta_t[j] = delta_t[j] - delta_t.front();
        }
        delta_t.front() = 0;

        int factor = 1;
        A.row(0) = Eigen::MatrixXd::Ones(1, matrix_dim);
        for (size_t j = 1; j < matrix_dim; ++j) {  // row
            factor = factor * j;
            Eigen::MatrixXd tmp_row(1, matrix_dim);
            for (size_t k = 0; k < matrix_dim; ++k) {
                tmp_row(0, k) = pow(delta_t[k], j) / factor;  // col
            }
            A.row(j) = tmp_row;
        }

        b(1) = 1;
        if (A.determinant() == 0) {
            std::cerr << "Matrix A is not invertible";
            return false;
        }
        matrix_coeff_vec->at(n).block(0, 0, matrix_dim, 1) =
            A.colPivHouseholderQr().solve(b);
        n = n + 1;
    }
    return true;
}

void SpeedMpcSolver::GetConfig(MpcOsqp::Input* input) const {
    // dimensions
    input->state_dim = state_dim_;
    input->control_dim = control_dim_;
    input->state_constraint_dim = state_constraint_dim_;
    input->control_constraint_dim = control_constraint_dim_;
    input->state_cost_function_dim = state_cost_function_dim_;
    input->horizon = horizon_step_;
    input->max_iter = 200;
    input->eps_abs = 5e-4;
    input->eps_rel = 5e-4;
    input->eps_prim_inf = 5e-4;
    return;
}

void SpeedMpcSolver::GetSolution(
    std::vector<SpeedOptimizeState>* states,
    std::vector<SpeedOptimizeControl>* controls) const {
    states->clear();
    states->reserve(horizon_step_ + 1);
    for (size_t i = 0; i < horizon_step_ + 1; ++i) {
        states->emplace_back(SpeedOptimizeState(
            solution_.at(i * state_dim_), solution_.at(i * state_dim_ + 1),
            solution_.at(i * state_dim_ + 2)));
    }

    controls->clear();
    controls->reserve(horizon_step_);
    for (size_t i = 0; i < horizon_step_; ++i) {
        controls->emplace_back(SpeedOptimizeControl(
            solution_.at(i + state_dim_ * (horizon_step_ + 1))));
    }
}
}  // namespace camera
}  // namespace perception
}  // namespace senseAD
