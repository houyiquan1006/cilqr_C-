/*
 * Copyright (C) 2022-2023 by SenseTime Group Limited. All rights reserved.
 * Bowen Zhang <zhangbowen2@senseauto.com>
 */
#include "cilqr_optimizer/cilqr_constructor.hpp"
#include "cilqr_optimizer/cilqr_utils.hpp"
#include <cmath>

Status CilqrConstructor::Init(
    CilqrConfig* cfg,
    const std::vector<PredictionObject>& model_obstacles) {
    horizon_ = cfg->ilqr.horizon;
    dense_horizon_ = cfg->ilqr.dense_horizon;
    num_states_ = cfg->ilqr.num_states;
    num_ctrls_ = cfg->ilqr.num_ctrls;
    num_auxiliary_ = cfg->ilqr.num_auxiliary;
    wheelbase_ = cfg->ego_vehicle.wheel_base;
    boundary_safe_ = cfg->constraint.boundary_safe;
    state_cost_ = Eigen::MatrixXd::Zero(num_states_, num_states_);
    control_cost_ = Eigen::MatrixXd::Zero(num_ctrls_, num_ctrls_);
    auxiliary_cost_ = Eigen::MatrixXd::Zero(num_auxiliary_, num_auxiliary_);
    q1_states_ = Eigen::MatrixXd::Zero(num_states_, 1);
    q2_states_ = Eigen::MatrixXd::Zero(num_states_, 1);
    q1_controls_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    q2_controls_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    state_constraints_min_ = Eigen::MatrixXd::Zero(num_states_, 1);
    state_constraints_max_ = Eigen::MatrixXd::Zero(num_states_, 1);
    control_constraints_min_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    control_constraints_max_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);

    control_regularization_cost_ =
        Eigen::MatrixXd::Zero(num_ctrls_, num_ctrls_);
    state_regularization_cost_ =
        Eigen::MatrixXd::Zero(num_states_, num_states_);

    for (int i = 0; i < num_states_; i++) {
        state_cost_(i, i) = cfg->cost_weight.w_states[i];
    }
    for (int i = 0; i < num_ctrls_; i++) {
        control_cost_(i, i) = cfg->cost_weight.w_controls[i];
    }
    for (int i = 0; i < num_auxiliary_; i++) {
        auxiliary_cost_(i, i) = cfg->cost_weight.w_auxiliary[i];
    }
    for (int i = 0; i < num_states_; i++) {
        q1_states_(i, 0) = cfg->cost_weight.q1_states[i];
        q2_states_(i, 0) = cfg->cost_weight.q2_states[i];
    }
    for (int i = 0; i < num_ctrls_; i++) {
        q1_controls_(i, 0) = cfg->cost_weight.q1_controls[i];
        q2_controls_(i, 0) = cfg->cost_weight.q2_controls[i];
    }
    for (int i = 0; i < num_states_; i++) {
        state_constraints_min_(i, 0) = cfg->constraint.state_constraints_min[i];
        state_constraints_max_(i, 0) = cfg->constraint.state_constraints_max[i];
    }
    for (int i = 0; i < num_ctrls_; i++) {
        control_constraints_min_(i, 0) =
            cfg->constraint.control_constraints_min[i];
        control_constraints_max_(i, 0) =
            cfg->constraint.control_constraints_max[i];
    }
    mul_frame_state_constraints_min_.clear();
    mul_frame_state_constraints_max_.clear();
    for (int j = 0; j < cfg->ilqr.horizon; j++) {
        mul_frame_state_constraints_min_.emplace_back(state_constraints_min_);
        mul_frame_state_constraints_max_.emplace_back(state_constraints_max_);
    }
    // Identity matrix indicating which channel is activated
    P_barrier_states_ = Eigen::MatrixXd::Zero(num_states_, 1);
    P_barrier_controls_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    P_states_ = Eigen::MatrixXd::Zero(num_states_, num_states_);
    P_controls_ = Eigen::MatrixXd::Zero(num_ctrls_, num_states_);
    for (int i = 0; i < num_states_; i++) {
        P_barrier_states_(i, 0) =
            cfg->cost_weight.q1_states[i] == 0.0 ? 0.0 : 1.0;
        P_states_(i, i) = cfg->cost_weight.w_states[i] == 0.0 ? 0.0 : 1.0;
    }
    for (int i = 0; i < num_ctrls_; i++) {
        P_barrier_controls_(i, 0) =
            cfg->cost_weight.q1_controls[i] == 0.0 ? 0.0 : 1.0;
        P_controls_(i, i) = cfg->cost_weight.w_controls[i] == 0.0 ? 0.0 : 1.0;
    }
    q1_boundary_ = cfg->cost_weight.q1_boundary;
    q2_boundary_ = cfg->cost_weight.q2_boundary;
    q1_front_ = cfg->cost_weight.q1_front;
    q2_front_ = cfg->cost_weight.q2_front;
    q1_rear_ = cfg->cost_weight.q1_rear;
    q2_rear_ = cfg->cost_weight.q2_rear;

    // target_trajectory_start_index_ = 0;
    target_trajectory_ = Eigen::MatrixXd::Zero(num_states_, horizon_);
    // target_last_trajectory_start_index_ = 0;
    target_last_trajectory_ = Eigen::MatrixXd::Zero(num_states_, horizon_);
    // set obstacles avoidance constraint
    setObsConstraints(*cfg, model_obstacles);

    return Status::SUCCESS;
}

void CilqrConstructor::setObsConstraints(
    const CilqrConfig& cfg,
    const std::vector<PredictionObject>& model_obstacles) {
    obs_constraints_.clear();
    if (model_obstacles.empty()) {
        return;
    }
    // TODO(BOWEN): support using obstacle constraint
    for (auto& pop : model_obstacles) {
        // if (pop.polygon_contour.size() < 3) {
        //     std::cout << "model obstacle polygon size less than 3, id: "
        //                  << pop.id;
        //     continue;
        // }
        // std::vector<cv::Point2f> contour;
        // if (pop.polygon_contour.size() >= 6) {
        //     contour = pop.polygon_contour;
        // } else {
        //     for (int i = 1; i < pop.polygon_contour.size(); i++) {
        //         contour.push_back(pop.polygon_contour[i - 1]);
        //         cv::Point2f mid =
        //             pop.polygon_contour[i] + pop.polygon_contour[i - 1];
        //         contour.push_back(cv::Point2f(mid.x / 2, mid.y / 2));
        //         contour.push_back(pop.polygon_contour[i]);
        //     }
        // }
        // cv::RotatedRect box = cv::fitEllipse(contour);
        // contour.clear();

        // CilqrObstacle obs(box.size.width, box.size.height);
        // CilqrObstacle obs(box.size.height, box.size.width);
        CilqrObstacle obs(pop.length, pop.width);
        obs.Init(cfg);
        std::vector<std::vector<double>> npc_states;

        cv::Point2f obs_positon_end;
        cv::Point2f obs_direction_end;
        cv::Point2f obs_speed_end;
        for (auto& traj : pop.trajectory_array) {
            if (traj.trajectory_point_array.size() == 0) {
                continue;
            }
            obs_speed_end.x = traj.trajectory_point_array.back().speed.x;
            obs_speed_end.y = traj.trajectory_point_array.back().speed.y;

            obs_positon_end.x = traj.trajectory_point_array.back().position.x;
            obs_positon_end.y = traj.trajectory_point_array.back().position.y;

            obs_direction_end.x =
                traj.trajectory_point_array.back().direction.x;
            obs_direction_end.y =
                traj.trajectory_point_array.back().direction.y;

            for (int i = 0; i < horizon_ + 1; i++) {
                std::vector<double> one_state(num_states_);
                if (i < traj.trajectory_point_array.size()) {
                    one_state[0] = traj.trajectory_point_array[i].position.x;
                    one_state[1] = traj.trajectory_point_array[i].position.y;
                    one_state[2] =
                        std::sqrt(traj.trajectory_point_array[i].speed.x *
                                      traj.trajectory_point_array[i].speed.x +
                                  traj.trajectory_point_array[i].speed.y *
                                      traj.trajectory_point_array[i].speed.y);
                    one_state[3] =
                        std::atan2(traj.trajectory_point_array[i].direction.y,
                                   traj.trajectory_point_array[i].direction.x);
                    one_state[4] = 0.0;
                    one_state[5] = 0.0;
                    npc_states.emplace_back(one_state);
                    std::cout << "i: " << i << " x: " << one_state[0] << " y: " << one_state[1] << " v: " << one_state[2] << " theta: " << one_state[3] << std::endl;

                }
                // else {
                //     one_state[0] = obs_positon_end.x +
                //                 obs_speed_end.x * 0.2 *
                //                     (i - traj.trajectory_point_array.size() +
                //                     1);
                //     one_state[1] = obs_positon_end.y +
                //                 obs_speed_end.y * 0.2 *
                //                     (i - traj.trajectory_point_array.size() +
                //                     1);
                //     one_state[2] = std::sqrt(obs_speed_end.x *
                //     obs_speed_end.x +
                //                             obs_speed_end.y *
                //                             obs_speed_end.y);
                //     one_state[3] =
                //         std::atan2(obs_direction_end.y, obs_direction_end.x);
                //     one_state[4] = 0.0;
                //     npc_states.emplace_back(one_state);
                // }
            }
            obs.setNpcStates(npc_states);
            obs.setTransformMatrix();
            obs_constraints_.emplace_back(obs);
        }
    }
}

void CilqrConstructor::findRearBoundaryStartIndex(
    const std::vector<std::vector<BoundaryPoint>>& boundary,
    const Eigen::MatrixXd& X) {
    rear_closest_boundary_start_index_.clear();
    for (int j = 0; j < boundary.size(); j++) {
        int index = 0;
        double min_distance = std::numeric_limits<double>::max();
        for (int i = 0; i < boundary[j].size(); i++) {
            double cur_distance;
            cur_distance = pow(X(0, 0) - boundary[j][i].pos.x, 2) +
                           pow(X(1, 0) - boundary[j][i].pos.y, 2);
            if (cur_distance < min_distance) {
                index = i;
                min_distance = cur_distance;
            }
        }
        rear_closest_boundary_start_index_.emplace_back(index);
        rear_closest_boundary_distances_.emplace_back(min_distance);
    }
}

void CilqrConstructor::findFrontBoundaryStartIndex(
    const std::vector<std::vector<BoundaryPoint>>& boundary,
    const Eigen::MatrixXd& X) {
    front_closest_boundary_start_index_.clear();
    for (int j = 0; j < boundary.size(); j++) {
        int index = 0;
        double min_distance = std::numeric_limits<double>::max();
        for (int i = 0; i < boundary[j].size(); i++) {
            double cur_distance;
            cur_distance = pow(X(0, 0) - boundary[j][i].pos.x, 2) +
                           pow(X(1, 0) - boundary[j][i].pos.y, 2);
            if (cur_distance < min_distance) {
                index = i;
                min_distance = cur_distance;
            }
        }
        front_closest_boundary_start_index_.emplace_back(index);
    }
}

void CilqrConstructor::findRearClosestBoundary(
    const std::vector<std::vector<BoundaryPoint>>& boundary,
    const Eigen::MatrixXd& X_seq) {
    rear_closest_boundary_.clear();
    rear_closest_boundary_direction_.clear();
    for (int j = 0; j < boundary.size(); j++) {
        Eigen::MatrixXd closest_boundary =
            Eigen::MatrixXd::Zero(num_states_, horizon_);
        Eigen::MatrixXd closest_boundary_direction =
            Eigen::MatrixXd::Zero(static_cast<int>(3), horizon_);
        int count = rear_closest_boundary_start_index_[j];
        for (int i = 0; i < horizon_; i++) {
            if (count > boundary[j].size() - 1) {
                count = boundary[j].size() - 1;
                closest_boundary(0, i) = boundary[j][count].pos.x;
                closest_boundary(1, i) = boundary[j][count].pos.y;
                closest_boundary_direction(0, i) = boundary[j][count].dir.x;
                closest_boundary_direction(1, i) = boundary[j][count].dir.y;
                if (-boundary[j][count].dir.y *
                            (X_seq(0, 0) - boundary[j][count].pos.x) +
                        boundary[j][count].dir.x *
                            (X_seq(1, 0) - boundary[j][count].pos.y) <
                    0) {
                    closest_boundary_direction(2, i) = 1.0;
                } else {
                    closest_boundary_direction(2, i) = -1.0;
                }
                // if (boundary[j][count].boundary_type ==
                //     BoundaryDirection::LEFT) {
                //     closest_boundary_direction(2, i) = 1.0;
                // } else if (boundary[j][count].boundary_type ==
                //            BoundaryDirection::RIGHT) {
                //     closest_boundary_direction(2, i) = -1.0;
                // }
                count++;
                continue;
            }
            double cur_distance;
            cur_distance = pow(X_seq(0, i + 1) - boundary[j][count].pos.x, 2) +
                           pow(X_seq(1, i + 1) - boundary[j][count].pos.y, 2);
            double next_distance;
            next_distance =
                pow(X_seq(0, i + 1) - boundary[j][count + 1].pos.x, 2) +
                pow(X_seq(1, i + 1) - boundary[j][count + 1].pos.y, 2);
            while ((next_distance < cur_distance) &&
                   ((count + 1) < boundary[j].size())) {
                count++;
                cur_distance = next_distance;
                next_distance =
                    pow(X_seq(0, i + 1) - boundary[j][count + 1].pos.x, 2) +
                    pow(X_seq(1, i + 1) - boundary[j][count + 1].pos.y, 2);
            }
            closest_boundary(0, i) = boundary[j][count].pos.x;
            closest_boundary(1, i) = boundary[j][count].pos.y;
            closest_boundary_direction(0, i) = boundary[j][count].dir.x;
            closest_boundary_direction(1, i) = boundary[j][count].dir.y;
            if (-boundary[j][count].dir.y *
                        (X_seq(0, 0) - boundary[j][count].pos.x) +
                    boundary[j][count].dir.x *
                        (X_seq(1, 0) - boundary[j][count].pos.y) <
                0) {
                closest_boundary_direction(2, i) = 1.0;
            } else {
                closest_boundary_direction(2, i) = -1.0;
            }
            // if (boundary[j][count].boundary_type == BoundaryDirection::LEFT)
            // {
            //     closest_boundary_direction(2, i) = 1.0;
            // } else if (boundary[j][count].boundary_type ==
            //            BoundaryDirection::RIGHT) {
            //     closest_boundary_direction(2, i) = -1.0;
            // }
        }
        rear_closest_boundary_.emplace_back(closest_boundary);
        rear_closest_boundary_direction_.emplace_back(
            closest_boundary_direction);
    }
}

void CilqrConstructor::findFrontClosestBoundary(
    const std::vector<std::vector<BoundaryPoint>>& boundary,
    const Eigen::MatrixXd& X_seq) {
    front_closest_boundary_.clear();
    front_closest_boundary_direction_.clear();
    for (int j = 0; j < boundary.size(); j++) {
        Eigen::MatrixXd closest_boundary =
            Eigen::MatrixXd::Zero(num_states_, horizon_);
        Eigen::MatrixXd closest_boundary_direction =
            Eigen::MatrixXd::Zero(static_cast<int>(3), horizon_);
        int count = front_closest_boundary_start_index_[j];
        for (int i = 0; i < horizon_; i++) {
            if (count > boundary[j].size() - 1) {
                count = boundary[j].size() - 1;
                closest_boundary(0, i) = boundary[j][count].pos.x;
                closest_boundary(1, i) = boundary[j][count].pos.y;
                closest_boundary_direction(0, i) = boundary[j][count].dir.x;
                closest_boundary_direction(1, i) = boundary[j][count].dir.y;
                if (-boundary[j][count].dir.y *
                            (X_seq(0, 0) - boundary[j][count].pos.x) +
                        boundary[j][count].dir.x *
                            (X_seq(1, 0) - boundary[j][count].pos.y) <
                    0) {
                    closest_boundary_direction(2, i) = 1.0;
                } else {
                    closest_boundary_direction(2, i) = -1.0;
                }
                // if (boundary[j][count].boundary_type ==
                //     BoundaryDirection::LEFT) {
                //     closest_boundary_direction(2, i) = 1.0;
                // } else if (boundary[j][count].boundary_type ==
                //            BoundaryDirection::RIGHT) {
                //     closest_boundary_direction(2, i) = -1.0;
                // }
                count++;
                continue;
            }
            double cur_distance;
            cur_distance = pow(X_seq(0, i + 1) - boundary[j][count].pos.x, 2) +
                           pow(X_seq(1, i + 1) - boundary[j][count].pos.y, 2);
            double next_distance;
            next_distance =
                pow(X_seq(0, i + 1) - boundary[j][count + 1].pos.x, 2) +
                pow(X_seq(1, i + 1) - boundary[j][count + 1].pos.y, 2);
            while ((next_distance < cur_distance) &&
                   ((count + 1) < boundary[j].size())) {
                count++;
                cur_distance = next_distance;
                next_distance =
                    pow(X_seq(0, i + 1) - boundary[j][count + 1].pos.x, 2) +
                    pow(X_seq(1, i + 1) - boundary[j][count + 1].pos.y, 2);
            }
            closest_boundary(0, i) = boundary[j][count].pos.x;
            closest_boundary(1, i) = boundary[j][count].pos.y;
            closest_boundary_direction(0, i) = boundary[j][count].dir.x;
            closest_boundary_direction(1, i) = boundary[j][count].dir.y;
            if (-boundary[j][count].dir.y *
                        (X_seq(0, 0) - boundary[j][count].pos.x) +
                    boundary[j][count].dir.x *
                        (X_seq(1, 0) - boundary[j][count].pos.y) <
                0) {
                closest_boundary_direction(2, i) = 1.0;
            } else {
                closest_boundary_direction(2, i) = -1.0;
            }
            // if (boundary[j][count].boundary_type == BoundaryDirection::LEFT)
            // {
            //     closest_boundary_direction(2, i) = 1.0;
            // } else if (boundary[j][count].boundary_type ==
            //            BoundaryDirection::RIGHT) {
            //     closest_boundary_direction(2, i) = -1.0;
            // }
        }
        front_closest_boundary_.emplace_back(closest_boundary);
        front_closest_boundary_direction_.emplace_back(
            closest_boundary_direction);
    }
}

double CilqrConstructor::barrierFunction(const double q1,
                                         const double q2,
                                         const double c) {
    return q1 * exp(q2 * c);
}
double CilqrConstructor::barrierFunction(const Eigen::MatrixXd& q1,
                                         const Eigen::MatrixXd& q2,
                                         const Eigen::MatrixXd& c) {
    return double(
        (q1.transpose() * (q2.array() * c.array()).exp().matrix())(0, 0));
}
Eigen::MatrixXd CilqrConstructor::barrierFunctionGrad(
    const double q1,
    const double q2,
    const double c,
    const Eigen::MatrixXd& c_grad) {
    Eigen::MatrixXd result;
    result = q1 * q2 * exp(q2 * c) * c_grad;
    return result;
}
Eigen::MatrixXd CilqrConstructor::barrierFunctionGrad(
    const Eigen::MatrixXd& q1,
    const Eigen::MatrixXd& q2,
    const Eigen::MatrixXd& c,
    const Eigen::MatrixXd& c_grad) {
    Eigen::MatrixXd result;
    result = (q1.array() * q2.array() * (q2.array() * c.array()).exp() *
              c_grad.array())
                 .matrix();
    return result;
}
Eigen::MatrixXd CilqrConstructor::barrierFunctionHessian(
    const double q1,
    const double q2,
    const double c,
    const Eigen::MatrixXd& c_grad) {
    Eigen::MatrixXd result;
    result = q1 * q2 * q2 * exp(q2 * c) * c_grad * c_grad.transpose();
    return result;
}

Eigen::MatrixXd CilqrConstructor::barrierFunctionHessian(
    const Eigen::MatrixXd& q1,
    const Eigen::MatrixXd& q2,
    const Eigen::MatrixXd& c,
    const Eigen::MatrixXd& c_grad) {
    Eigen::MatrixXd result;
    result =
        (q1.array() * q2.array() * q2.array() * (q2.array() * c.array()).exp())
            .matrix()
            .asDiagonal();
    result *= c_grad.asDiagonal();
    return result;
}
Eigen::MatrixXd CilqrConstructor::getStateCostGrad(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& X_last, const int i) {
    Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
    StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);
    // X_front(0, 0) = X(0, 0) + cos(X(2, 0)) * wheelbase_;
    // X_front(1, 0) = X(1, 0) + sin(X(2, 0)) * wheelbase_;
    // for(int i = 2 ; i < num_states_ ; i++){
    //     X_front(i, 0) = X(i, 0);
    // }

    // state tracking error cost
    Eigen::MatrixXd state_cost_grad;
    state_cost_grad = Eigen::MatrixXd::Zero(num_states_, 1);
    state_cost_grad =
        state_cost_grad + 2 * state_cost_ * (X - target_trajectory_.col(i - 1));
    // state regularization cost
    if (!target_last_trajectory_.isZero()) {
        state_cost_grad =
            state_cost_grad + 2 * state_regularization_cost_ *
                                  (X - target_last_trajectory_.col(i - 1));
    }
    // kappa constraint barrier function cost

    Eigen::MatrixXd c3 = X - mul_frame_state_constraints_max_.at(i - 1);
    Eigen::MatrixXd c4 = mul_frame_state_constraints_min_.at(i - 1) - X;
    Eigen::MatrixXd barrier_grad3;
    Eigen::MatrixXd barrier_grad4;
    // double c3 = X(3, 0) - state_constraints_max_[3];
    barrier_grad3 =
        barrierFunctionGrad(q1_states_, q2_states_, c3, P_barrier_states_);
    // double c4 = state_constraints_min_[3] - X(3, 0);
    barrier_grad4 =
        barrierFunctionGrad(q1_states_, q2_states_, c4, -P_barrier_states_);
    state_cost_grad = state_cost_grad + barrier_grad3;
    state_cost_grad = state_cost_grad + barrier_grad4;

    // for (int j = 0; j < rear_closest_boundary_.size(); j++) {
    //     state_cost_grad =
    //         state_cost_grad + getBoundaryGrad(q1_boundary_, q2_boundary_, X,
    //         i,
    //                                           j, rear_closest_boundary_,
    //                                           rear_closest_boundary_direction_);
    // }

    for (int j = 0; j < rear_closest_boundary_segment_.size(); j++) {
        // Get projected line segment of state Xi
        Eigen::MatrixXd closest_boundary_segment =
            rear_closest_boundary_segment_[j].col(i - 1);

        // Get projected line segment tangential vectors of state Xi
        Eigen::MatrixXd closest_boundary_segment_direction =
            rear_closest_boundary_segment_direction_[j].col(i - 1);

        // Calculate cost
        state_cost_grad =
            state_cost_grad + getSignedDistanceBoundaryGrad(
                                  q1_boundary_, q2_boundary_, X,
                                  closest_boundary_segment,
                                  closest_boundary_segment_direction, "rear");
    }

    // for (int j = 0; j < front_closest_boundary_.size(); j++) {
    //     state_cost_grad = state_cost_grad +
    //                       getBoundaryGrad(q1_boundary_, q2_boundary_,
    //                       X_front,
    //                                       i, j, front_closest_boundary_,
    //                                       front_closest_boundary_direction_);
    // }

    for (int j = 0; j < front_closest_boundary_.size(); j++) {
        // Get projected line segment of state Xi
        Eigen::MatrixXd closest_boundary_segment =
            front_closest_boundary_segment_[j].col(i - 1);

        // Get projected line segment tangential vectors of state Xi
        Eigen::MatrixXd closest_boundary_segment_direction =
            front_closest_boundary_segment_direction_[j].col(i - 1);

        // Calculate cost
        state_cost_grad =
            state_cost_grad + getSignedDistanceBoundaryGrad(
                                  q1_boundary_, q2_boundary_, X_front,
                                  closest_boundary_segment,
                                  closest_boundary_segment_direction, "front");
    }

    for (int j = 0; j < obs_constraints_.size(); j++) {
        if (i >= obs_constraints_[j].getNpcStatesSize()) continue;
        // set front constraint
        Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
        StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);
        state_cost_grad =
            state_cost_grad + obs_constraints_[j].getObstacleGrad(
                                  q1_front_, q2_front_, X_front, i);
        // set rear constraint
        state_cost_grad = state_cost_grad + obs_constraints_[j].getObstacleGrad(
                                                q1_rear_, q2_rear_, X, i);
    }

    return state_cost_grad;
}

Eigen::MatrixXd CilqrConstructor::getStateCostHessian(const Eigen::MatrixXd& X,
                                                      const int i) {
    Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
    StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);
    // X_front(0, 0) = X(0, 0) + cos(X(2, 0)) * wheelbase_;
    // X_front(1, 0) = X(1, 0) + sin(X(2, 0)) * wheelbase_;
    // for(int i = 2 ; i < num_states_ ; i++){
    //     X_front(i, 0) = X(i, 0);
    // }
    // state tracking error cost
    Eigen::MatrixXd state_cost_hessian;
    state_cost_hessian = Eigen::MatrixXd::Zero(num_states_, num_states_);
    state_cost_hessian = state_cost_hessian + 2 * state_cost_;
    state_cost_hessian = state_cost_hessian + 2 * state_regularization_cost_;
    // constraint barrier function cost
    Eigen::MatrixXd barrier_hessian3;
    Eigen::MatrixXd barrier_hessian4;
    Eigen::MatrixXd c3 = X - mul_frame_state_constraints_max_.at(i - 1);
    Eigen::MatrixXd c4 = mul_frame_state_constraints_min_.at(i - 1) - X;
    barrier_hessian3 =
        barrierFunctionHessian(q1_states_, q2_states_, c3, P_barrier_states_);
    barrier_hessian4 =
        barrierFunctionHessian(q1_states_, q2_states_, c4, P_barrier_states_);
    state_cost_hessian = state_cost_hessian + barrier_hessian3;
    state_cost_hessian = state_cost_hessian + barrier_hessian4;

    // for (int j = 0; j < rear_closest_boundary_.size(); j++) {
    //     state_cost_hessian =
    //         state_cost_hessian +
    //         getBoundaryHessian(q1_boundary_, q2_boundary_, X, i, j,
    //                            rear_closest_boundary_,
    //                            rear_closest_boundary_direction_);
    // }

    // Traverse every boundary and calculate boundary cost hessian of rear axle
    for (int j = 0; j < rear_closest_boundary_segment_.size(); j++) {
        // Get projected line segment of state Xi
        Eigen::MatrixXd closest_boundary_segment =
            rear_closest_boundary_segment_[j].col(i - 1);

        // Get projected line segment tangential vectors of state Xi
        Eigen::MatrixXd closest_boundary_segment_direction =
            rear_closest_boundary_segment_direction_[j].col(i - 1);

        // Calculate cost
        state_cost_hessian =
            state_cost_hessian +
            getSignedDistanceBoundaryHessian(
                q1_boundary_, q2_boundary_, X, closest_boundary_segment,
                closest_boundary_segment_direction, "rear");
    }

    // Traverse every boundary and calculate boundary cost hessian of rear axle
    for (int j = 0; j < front_closest_boundary_segment_.size(); j++) {
        // Get projected line segment of state Xi
        Eigen::MatrixXd closest_boundary_segment =
            front_closest_boundary_segment_[j].col(i - 1);

        // Get projected line segment tangential vectors of state Xi
        Eigen::MatrixXd closest_boundary_segment_direction =
            front_closest_boundary_segment_direction_[j].col(i - 1);

        // Calculate cost
        state_cost_hessian =
            state_cost_hessian +
            getSignedDistanceBoundaryHessian(
                q1_boundary_, q2_boundary_, X_front, closest_boundary_segment,
                closest_boundary_segment_direction, "front");
    }

    // for (int j = 0; j < front_closest_boundary_.size(); j++) {
    //     state_cost_hessian =
    //         state_cost_hessian +
    //         getBoundaryHessian(q1_boundary_, q2_boundary_, X_front, i, j,
    //                            front_closest_boundary_,
    //                            front_closest_boundary_direction_);
    // }

    for (int j = 0; j < obs_constraints_.size(); j++) {
        if (i >= obs_constraints_[j].getNpcStatesSize()) continue;
        Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
        StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);
        state_cost_hessian =
            state_cost_hessian + obs_constraints_[j].getObstacleHessian(
                                     q1_front_, q2_front_, X_front, i);
        state_cost_hessian =
            state_cost_hessian +
            obs_constraints_[j].getObstacleHessian(q1_rear_, q2_rear_, X, i);
    }
    return state_cost_hessian;
}

Eigen::MatrixXd CilqrConstructor::getControlCostGrad(
    const Eigen::MatrixXd& U, const Eigen::MatrixXd& U_last) {
    // control cost
    Eigen::MatrixXd control_cost_grad;
    control_cost_grad = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    control_cost_grad = control_cost_grad + 2 * control_cost_ * U;

    // control regularization cost
    Eigen::MatrixXd control_regularization_cost_grad =
        Eigen::MatrixXd::Zero(num_ctrls_, 1);
    Eigen::MatrixXd delta_U;
    delta_U = U - U_last;
    control_regularization_cost_grad =
        control_regularization_cost_grad +
        2 * control_regularization_cost_ * delta_U;
    control_cost_grad = control_cost_grad + control_regularization_cost_grad;
    // constraint barrier function cost
    Eigen::MatrixXd barrier_grad1;
    Eigen::MatrixXd barrier_grad2;

    Eigen::MatrixXd c1 = U - control_constraints_max_;
    Eigen::MatrixXd c2 = control_constraints_min_ - U;

    barrier_grad1 = barrierFunctionGrad(q1_controls_, q2_controls_, c1,
                                        P_barrier_controls_);
    barrier_grad2 = barrierFunctionGrad(q1_controls_, q2_controls_, c2,
                                        -P_barrier_controls_);

    control_cost_grad = control_cost_grad + barrier_grad1;
    control_cost_grad = control_cost_grad + barrier_grad2;

    return control_cost_grad;
}

Eigen::MatrixXd CilqrConstructor::getControlCostHessian(
    const Eigen::MatrixXd& U) {
    // control cost
    Eigen::MatrixXd control_cost_hessian;
    control_cost_hessian = Eigen::MatrixXd::Zero(num_ctrls_, num_ctrls_);
    control_cost_hessian = control_cost_hessian + 2 * control_cost_ +
                           2 * control_regularization_cost_;

    // kapparate constraint barrier function cost
    Eigen::MatrixXd barrier_hessian1;
    Eigen::MatrixXd barrier_hessian2;

    Eigen::MatrixXd c1 = U - control_constraints_max_;
    Eigen::MatrixXd c2 = control_constraints_min_ - U;
    barrier_hessian1 = barrierFunctionHessian(q1_controls_, q2_controls_, c1,
                                              P_barrier_controls_);
    barrier_hessian2 = barrierFunctionHessian(q1_controls_, q2_controls_, c2,
                                              P_barrier_controls_);
    control_cost_hessian = control_cost_hessian + barrier_hessian1;
    control_cost_hessian = control_cost_hessian + barrier_hessian2;

    return control_cost_hessian;
}

double CilqrConstructor::getBoundaryCost(
    const double q1,
    const double q2,
    const Eigen::MatrixXd& X,
    const int i,
    const int j,
    const std::vector<Eigen::MatrixXd>& closest_boundary,
    const std::vector<Eigen::MatrixXd>& closest_boundary_direction) {
    Eigen::MatrixXd tmp;
    tmp = (X - closest_boundary[j].col(i - 1)).transpose() * P_states_ *
          (X - closest_boundary[j].col(i - 1));
    // TODO(ZBW) pass boundary safe via real half width of lane
    // double c = boundary_safe_ * boundary_safe_ - tmp(0, 0);
    double c = boundary_safe_ * boundary_safe_ - tmp(0, 0);
    double boundary_cost = q1 * exp(q2 * c);
    // Eigen::MatrixXd T = Eigen::MatrixXd::Zero(1, num_states_);
    // T(0, 0) = -closest_boundary_direction[j](1, i - 1);
    // T(0, 1) = closest_boundary_direction[j](0, i - 1);
    // Eigen::MatrixXd tmp;
    // tmp = T * (X - closest_boundary[j].col(i - 1));
    // tmp(0, 0) =
    //     tmp(0, 0) + closest_boundary_direction[j](2, i - 1) * boundary_safe_;
    // double c = closest_boundary_direction[j](2, i - 1) * tmp(0, 0);
    // if (c > 0.0) {
    //     boundary_cost = q1 * exp(q2 * c) - q1 * q2 * c - q1;
    // }
    return boundary_cost;
}

Eigen::MatrixXd CilqrConstructor::getBoundaryGrad(
    const double q1,
    const double q2,
    const Eigen::MatrixXd& X,
    const int i,
    const int j_,
    const std::vector<Eigen::MatrixXd>& closest_boundary,
    const std::vector<Eigen::MatrixXd>& closest_boundary_direction) {
    Eigen::MatrixXd boundary_grad = Eigen::MatrixXd::Zero(num_states_, 1);
    int j = j_;
    if (j < 0 || j >= closest_boundary_direction.size()) {
        j = 0;
    }
    if (j >= closest_boundary_direction.size()) {
        j = closest_boundary_direction.size() - 1;
    }
    Eigen::MatrixXd tmp;
    tmp = (X - closest_boundary[j].col(i - 1)).transpose() * P_states_ *
          (X - closest_boundary[j].col(i - 1));
    double c = boundary_safe_ * boundary_safe_ - tmp(0, 0);
    boundary_grad = -2 * q1 * q2 * exp(q2 * c) * P_states_ *
                    (X - closest_boundary[j].col(i - 1));
    // Eigen::MatrixXd T = Eigen::MatrixXd::Zero(1, num_states_);
    // T(0, 0) = -closest_boundary_direction[j](1, i - 1);
    // T(0, 1) = closest_boundary_direction[j](0, i - 1);
    // Eigen::MatrixXd tmp;
    // tmp = T * (X - closest_boundary[j].col(i - 1));
    // tmp(0, 0) =
    //     tmp(0, 0) + closest_boundary_direction[j](2, i - 1) * boundary_safe_;
    // double c = closest_boundary_direction[j](2, i - 1) * tmp(0, 0);
    // if (c > 0.0)
    //     boundary_grad =
    //         closest_boundary_direction[j](2, i - 1) *
    //         (q1 * q2 * exp(q2 * c) * T.transpose() - q1 * q2 *
    //         T.transpose());
    return boundary_grad;
}

Eigen::MatrixXd CilqrConstructor::getBoundaryHessian(
    const double q1,
    const double q2,
    const Eigen::MatrixXd& X,
    const int i,
    const int j,
    const std::vector<Eigen::MatrixXd>& closest_boundary,
    const std::vector<Eigen::MatrixXd>& closest_boundary_direction) {
    Eigen::MatrixXd boundary_hessian =
        Eigen::MatrixXd::Zero(num_states_, num_states_);
    Eigen::MatrixXd tmp;
    tmp = (X - closest_boundary[j].col(i - 1)).transpose() * P_states_ *
          (X - closest_boundary[j].col(i - 1));
    double c = boundary_safe_ * boundary_safe_ - tmp(0, 0);
    boundary_hessian = 4 * q1 * q2 * q2 * exp(q2 * c) * P_states_ *
                           (X - closest_boundary[j].col(i - 1)) *
                           (X - closest_boundary[j].col(i - 1)).transpose() *
                           P_states_ -
                       2 * q1 * q2 * exp(q2 * c) * P_states_;
    // Eigen::MatrixXd T = Eigen::MatrixXd::Zero(1, num_states_);
    // T(0, 0) = -closest_boundary_direction[j](1, i - 1);
    // T(0, 1) = closest_boundary_direction[j](0, i - 1);
    // Eigen::MatrixXd tmp;
    // tmp = T * (X - closest_boundary[j].col(i - 1));
    // tmp(0, 0) =
    //     tmp(0, 0) + closest_boundary_direction[j](2, i - 1) * boundary_safe_;
    // double c = closest_boundary_direction[j](2, i - 1) * tmp(0, 0);
    // if (c > 0.0) {
    //     boundary_hessian = q1 * q2 * q2 * exp(q2 * c) * T.transpose() * T;
    // }
    return boundary_hessian;
}
double CilqrConstructor::getAuxiliaryCost(const Eigen::MatrixXd& X,
                                          const Eigen::MatrixXd& U,
                                          const int i) {
    // The time dim of input X and U are 1
    double auxiliary_cost = 0.0;
    Eigen::MatrixXd auxiliary_terms = Eigen::MatrixXd::Zero(num_auxiliary_, 1);
    double vel = X(2, 0);
    double lon_acc = X(4, 0);
    double yaw_rate = X(5, 0);
    double yaw_rate_dot = U(1, 0);
    auxiliary_terms(0, 0) = vel * yaw_rate;
    auxiliary_terms(1, 0) = lon_acc * yaw_rate + yaw_rate_dot * vel;
    auxiliary_cost = auxiliary_cost + (auxiliary_terms.transpose() *
                                       auxiliary_cost_ * auxiliary_terms)(0, 0);
    auxiliary_cost = (1 - static_cast<double>(i) / horizon_) * auxiliary_cost;

    // std::cout << "lat_acc auxiliary cost: " << (1 - static_cast<double>(i) / horizon_) * auxiliary_terms(0, 0) * auxiliary_cost_(0,0) * auxiliary_terms(0, 0) << std::endl;
    // std::cout << "lat_jerk auxiliary cost: " << (1 - static_cast<double>(i) / horizon_) * auxiliary_terms(1, 0) * auxiliary_cost_(1,1) * auxiliary_terms(1, 0) << std::endl;
    return auxiliary_cost;
}

Eigen::MatrixXd CilqrConstructor::getAuxiliaryCostGrad(const Eigen::MatrixXd& X,
                                                       const Eigen::MatrixXd& U,
                                                       const int i) {
    // The time dim of input X and U are 1
    Eigen::MatrixXd auxiliary_cost_grad;
    Eigen::MatrixXd auxiliary_weights;
    Eigen::MatrixXd auxiliary_terms;
    auxiliary_cost_grad = Eigen::MatrixXd::Zero(num_states_ + num_ctrls_, 1);
    auxiliary_weights = Eigen::MatrixXd::Zero(num_auxiliary_, 1);
    auxiliary_terms =
        Eigen::MatrixXd::Zero(num_states_ + num_ctrls_, num_auxiliary_);
    double vel = X(2, 0);
    double lon_acc = X(4, 0);
    double yaw_rate = X(5, 0);
    double yaw_rate_dot = U(1, 0);
    auxiliary_terms(2, 0) = vel * yaw_rate * yaw_rate;
    auxiliary_terms(2, 1) =
        (lon_acc * yaw_rate + yaw_rate_dot * vel) * yaw_rate_dot;
    auxiliary_terms(4, 1) =
        (lon_acc * yaw_rate + yaw_rate_dot * vel) * yaw_rate;
    auxiliary_terms(5, 0) = vel * vel * yaw_rate;
    auxiliary_terms(5, 1) = (lon_acc * yaw_rate + yaw_rate_dot * vel) * lon_acc;
    auxiliary_terms(7, 1) = (lon_acc * yaw_rate + yaw_rate_dot * vel) * vel;
    auxiliary_weights(0, 0) = auxiliary_cost_(0, 0);
    auxiliary_weights(1, 0) = auxiliary_cost_(1, 1);
    auxiliary_cost_grad =
        auxiliary_cost_grad + 2 * auxiliary_terms * auxiliary_weights;
    auxiliary_cost_grad = (1 - static_cast<double>(i) / horizon_) * auxiliary_cost_grad;
    return auxiliary_cost_grad;
}
Eigen::MatrixXd CilqrConstructor::getAuxiliaryCostHessian(
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& U, const int i) {
    Eigen::MatrixXd auxiliary_cost_hessian;
    Eigen::MatrixXd auxiliary_weights;
    Eigen::MatrixXd auxiliary_terms;
    auxiliary_cost_hessian = Eigen::MatrixXd::Zero(num_states_ + num_ctrls_,
                                                   num_states_ + num_ctrls_);
    auxiliary_weights = Eigen::MatrixXd::Zero(num_auxiliary_, 1);
    auxiliary_terms = Eigen::MatrixXd::Zero(num_states_ + num_ctrls_,
                                            num_states_ + num_ctrls_);
    double vel = X(2, 0);
    double lon_acc = X(4, 0);
    double yaw_rate = X(5, 0);
    double yaw_rate_dot = U(1, 0);
    auxiliary_terms(2, 2) = auxiliary_cost_(0, 0) * yaw_rate * yaw_rate +
                            auxiliary_cost_(1, 1) * yaw_rate_dot * yaw_rate_dot;
    auxiliary_terms(2, 4) = auxiliary_cost_(1, 1) * yaw_rate * yaw_rate_dot;
    auxiliary_terms(2, 5) = 2 * auxiliary_cost_(0, 0) * vel * yaw_rate +
                            auxiliary_cost_(1, 1) * lon_acc * yaw_rate_dot;
    auxiliary_terms(2, 7) =
        auxiliary_cost_(1, 1) * (lon_acc * yaw_rate + 2 * vel * yaw_rate_dot);

    auxiliary_terms(4, 2) = auxiliary_cost_(1, 1) * yaw_rate * yaw_rate_dot;
    auxiliary_terms(4, 4) = auxiliary_cost_(1, 1) * yaw_rate * yaw_rate;
    auxiliary_terms(4, 5) =
        auxiliary_cost_(1, 1) * (2 * lon_acc * yaw_rate + yaw_rate_dot * vel);
    auxiliary_terms(4, 7) = auxiliary_cost_(1, 1) * vel * yaw_rate;

    auxiliary_terms(5, 2) = 2 * auxiliary_cost_(0, 0) * vel * yaw_rate +
                            auxiliary_cost_(1, 1) * lon_acc * yaw_rate_dot;
    auxiliary_terms(5, 4) =
        auxiliary_cost_(1, 1) * (2 * lon_acc * yaw_rate + vel * yaw_rate_dot);
    auxiliary_terms(5, 5) = auxiliary_cost_(0, 0) * vel * vel +
                            auxiliary_cost_(1, 1) * lon_acc * lon_acc;
    auxiliary_terms(5, 7) = auxiliary_cost_(1, 1) * vel * lon_acc;
    auxiliary_terms(7, 2) =
        auxiliary_cost_(1, 1) * (lon_acc * yaw_rate + 2 * yaw_rate_dot * vel);
    auxiliary_terms(7, 4) = auxiliary_cost_(1, 1) * vel * yaw_rate;
    auxiliary_terms(7, 5) = auxiliary_cost_(1, 1) * lon_acc * vel;
    auxiliary_terms(7, 7) = auxiliary_cost_(1, 1) * vel * vel;
    auxiliary_cost_hessian = auxiliary_cost_hessian + 2 * auxiliary_terms;
    auxiliary_cost_hessian = (1 - static_cast<double>(i) / horizon_) * auxiliary_cost_hessian;
    return auxiliary_cost_hessian;
}
double CilqrConstructor::getStateCost(const Eigen::MatrixXd& X,
                                      const Eigen::MatrixXd& X_last,
                                      const int i) {
    double state_cost = 0;
    Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
    StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);

    // state tracking error cost
    Eigen::MatrixXd tmp;
    Eigen::MatrixXd delta_X;
    Eigen::MatrixXd X_reg;
    Eigen::MatrixXd X_regularization_cost;
    delta_X = X - target_trajectory_.col(i - 1);
    tmp = delta_X.transpose() * state_cost_ * delta_X;
    state_cost = state_cost + tmp(0, 0);
    if (!target_last_trajectory_.isZero()) {
        X_reg = X - target_last_trajectory_.col(i - 1);
        X_regularization_cost =
            X_reg.transpose() * state_regularization_cost_ * X_reg;
        state_cost = state_cost + X_regularization_cost(0, 0);

    }
    // constraint barrier function cost
    double total_state_barrier_cost{0.};
    Eigen::MatrixXd c1 = X - mul_frame_state_constraints_max_.at(i - 1);
    Eigen::MatrixXd c2 = mul_frame_state_constraints_min_.at(i - 1) - X;
    total_state_barrier_cost += barrierFunction(q1_states_, q2_states_, c1);
    total_state_barrier_cost += barrierFunction(q1_states_, q2_states_, c2);
    state_cost += total_state_barrier_cost;
    // for(int j = 0;j < num_states_; j++){
    //     double c1 = X(j, 0) - state_constraints_max_[j];
    //     total_state_barrier_cost = total_state_barrier_cost +
    //                             barrierFunction(q1_states_[j], q2_states_[j],
    //                             c1);
    //     double c2 = state_constraints_min_[j] - X(j, 0);
    //     total_state_barrier_cost = total_state_barrier_cost +
    //                            barrierFunction(q1_states_[j], q2_states_[j],
    //                            c2);
    // }

    // for (int j = 0; j < rear_closest_boundary_.size(); j++) {
    //     state_cost =
    //         state_cost + getBoundaryCost(q1_boundary_, q2_boundary_, X, i, j,
    //                                      rear_closest_boundary_,
    //                                      rear_closest_boundary_direction_);
    // }
    // Traverse every boundary and calculate boundary cost of rear axle
    for (int j = 0; j < rear_closest_boundary_segment_.size(); j++) {
        // Get projected line segment of state Xi
        Eigen::MatrixXd closest_boundary_segment =
            rear_closest_boundary_segment_[j].col(i - 1);

        // Get projected line segment tangential vectors of state Xi
        Eigen::MatrixXd closest_boundary_segment_direction =
            rear_closest_boundary_segment_direction_[j].col(i - 1);
        // Calculate cost
        state_cost = state_cost + getSignedDistanceBoundaryCost(
                                      q1_boundary_, q2_boundary_, X,
                                      closest_boundary_segment,
                                      closest_boundary_segment_direction);
    }
    for (int j = 0; j < front_closest_boundary_segment_.size(); j++) {
        // Get projected line segment of state Xi_front
        Eigen::MatrixXd closest_boundary_segment =
            front_closest_boundary_segment_[j].col(i - 1);

        // Get projected line segment tangential vectors of state Xi
        Eigen::MatrixXd closest_boundary_segment_direction =
            front_closest_boundary_segment_direction_[j].col(i - 1);

        // Calculate cost
        state_cost = state_cost + getSignedDistanceBoundaryCost(
                                      q1_boundary_, q2_boundary_, X_front,
                                      closest_boundary_segment,
                                      closest_boundary_segment_direction);
    }

    // for (int j = 0; j < front_closest_boundary_.size(); j++) {
    //     state_cost =
    //         state_cost + getBoundaryCost(q1_boundary_, q2_boundary_, X_front,
    //         i,
    //                                      j, front_closest_boundary_,
    //                                      front_closest_boundary_direction_);
    // }

    // double tmpcost = state_cost - total_state_barrier_cost - tmp(0, 0);
    for (int j = 0; j < obs_constraints_.size(); j++) {
        if (i >= obs_constraints_[j].getNpcStatesSize()) continue;
        // set front constraint
        Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
        StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);
        double tem_front_obs_cost = obs_constraints_[j].getObstacleCost(
            q1_front_, q2_front_, X_front, i);

        double tem_rear_obs_cost =
            obs_constraints_[j].getObstacleCost(q1_rear_, q2_rear_, X, i);

        state_cost = state_cost + obs_constraints_[j].getObstacleCost(
                                      q1_front_, q2_front_, X_front, i);
        // set rear constraint
        state_cost = state_cost + obs_constraints_[j].getObstacleCost(
                                      q1_rear_, q2_rear_, X, i);
        // std::cout << "rear obs cost at step " + std::to_string(i) + " : " << tem_rear_obs_cost << std::endl;
        // std::cout << "front obs cost at step " + std::to_string(i) + " : " << tem_front_obs_cost << std::endl;
    }
    return state_cost;
}

double CilqrConstructor::getControlCost(const Eigen::MatrixXd& U,
                                        const Eigen::MatrixXd& U_last) {
    double control_cost = 0;
    Eigen::MatrixXd c1 = U - control_constraints_max_;
    Eigen::MatrixXd c2 = control_constraints_min_ - U;
    control_cost += barrierFunction(q1_controls_, q2_controls_, c1);
    control_cost += barrierFunction(q1_controls_, q2_controls_, c2);

    // double c1 = U(0, 0) - kapparate_max_limit_;
    // control_cost =
    //     control_cost + barrierFunction(q1_kapparate_, q2_kapparate_, c1);
    // double c2 = kapparate_min_limit_ - U(0, 0);
    // control_cost =
    //     control_cost + barrierFunction(q1_kapparate_, q2_kapparate_, c2);

    Eigen::MatrixXd tmp;
    tmp = U.transpose() * control_cost_ * U;

    control_cost = control_cost + tmp(0, 0);
    // for (int i = 0; i < num_ctrls_; i++)
    //     std::cout << "control_cost " << i << "th dim: "
    //         <<  U(i, 0) * control_cost_(i, i) * U(i, 0) << std::endl;
    // std::cout << "control_cost: " << control_cost << std::endl;
    
    return control_cost;
}

std::tuple<double,double,double> CilqrConstructor::getTotalTraCost(const Eigen::MatrixXd& X_seq) {
    double total_tra_cost = 0.0;
    double total_x_cost = 0.0;
    double total_y_cost = 0.0;
    double total_vel_cost = 0.0;
    double total_yaw_cost = 0.0;
    double total_acc_cost = 0.0;
    double total_yaw_rate_cost = 0.0;
    for (int i = 0; i < horizon_; i++) {
        Eigen::MatrixXd tmp;
        Eigen::MatrixXd delta_X;
        delta_X = X_seq.col(i + 1) - target_trajectory_.col(i);
        tmp = delta_X.transpose() * state_cost_ * delta_X;
        total_x_cost += delta_X(0, 0) * delta_X(0, 0) * state_cost_(0, 0);
        total_y_cost += delta_X(1, 0) * delta_X(1, 0) * state_cost_(1, 1);
        total_vel_cost += delta_X(2, 0) * delta_X(2, 0) * state_cost_(2, 2);
        total_yaw_cost += delta_X(3, 0) * delta_X(3, 0) * state_cost_(3, 3);
        total_acc_cost += delta_X(4, 0) * delta_X(4, 0) * state_cost_(4, 4);
        total_yaw_rate_cost +=
            delta_X(5, 0) * delta_X(5, 0) * state_cost_(5, 5);
        total_tra_cost += tmp(0, 0);
    }

    return std::make_tuple(total_tra_cost, total_x_cost, total_y_cost);
}
double CilqrConstructor::getTotalStateRegCost(const Eigen::MatrixXd& X_seq) {
    double total_tra_cost = 0;
    if (target_last_trajectory_.isZero()) {
        return total_tra_cost;
    }
    for (int i = 0; i < horizon_; i++) {
        Eigen::MatrixXd tmp;
        Eigen::MatrixXd delta_X;
        delta_X = X_seq.col(i + 1) - target_last_trajectory_.col(i);
        tmp = delta_X.transpose() * state_regularization_cost_ * delta_X;
        total_tra_cost = total_tra_cost + tmp(0, 0);
    }
    return total_tra_cost;
}
double CilqrConstructor::getTotalBoundaryCost(const Eigen::MatrixXd& X_seq) {
    double total_boundary_cost = 0;
    for (int i = 0; i < horizon_; i++) {
        Eigen::MatrixXd X = X_seq.col(i + 1);
        Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, 1);
        StateTransitfromReartoFront(&X_front, X, 0, wheelbase_);

        for (int j = 0; j < rear_closest_boundary_segment_.size(); j++) {
            // Get projected line segment of state Xi
            Eigen::MatrixXd closest_boundary_segment =
                rear_closest_boundary_segment_[j].col(i);

            // Get projected line segment tangential vectors of state Xi
            Eigen::MatrixXd closest_boundary_segment_direction =
                rear_closest_boundary_segment_direction_[j].col(i);

            // Calculate Cost
            total_boundary_cost =
                total_boundary_cost + getSignedDistanceBoundaryCost(
                                          q1_boundary_, q2_boundary_, X,
                                          closest_boundary_segment,
                                          closest_boundary_segment_direction);
        }

        for (int j = 0; j < front_closest_boundary_segment_.size(); j++) {
            // Get projected line segment of state Xi
            Eigen::MatrixXd closest_boundary_segment =
                front_closest_boundary_segment_[j].col(i);

            // Get projected line segment tangential vectors of state Xi
            Eigen::MatrixXd closest_boundary_segment_direction =
                front_closest_boundary_segment_direction_[j].col(i);

            // Calculate Cost
            total_boundary_cost =
                total_boundary_cost + getSignedDistanceBoundaryCost(
                                          q1_boundary_, q2_boundary_, X_front,
                                          closest_boundary_segment,
                                          closest_boundary_segment_direction);
        }
    }

    return total_boundary_cost;
}

double CilqrConstructor::getTotalStateBarrierCost(
    const Eigen::MatrixXd& X_seq) {
    double total_state_barrier_cost = 0;
    for (int i = 0; i < horizon_; i++) {
        Eigen::MatrixXd X = X_seq.col(i + 1);
        Eigen::MatrixXd c1 = X - state_constraints_max_;
        Eigen::MatrixXd c2 = state_constraints_min_ - X;
        total_state_barrier_cost += barrierFunction(q1_states_, q2_states_, c1);
        total_state_barrier_cost += barrierFunction(q1_states_, q2_states_, c2);
        // for(int j = 0;j < num_states_; j++){
        //     double c1 = X(j, 0) - state_constraints_max_[j];
        //     total_state_barrier_cost = total_state_barrier_cost +
        //                             barrierFunction(q1_states_[j],
        //                             q2_states_[j], c1);
        //     double c2 = state_constraints_min_[j] - X(j, 0);
        //     total_state_barrier_cost = total_state_barrier_cost +
        //                            barrierFunction(q1_states_[j],
        //                            q2_states_[j], c2);
        // }
    }
    return total_state_barrier_cost;
}

void CilqrConstructor::getTotalCost(const Eigen::MatrixXd& X_seq,
                                    const Eigen::MatrixXd& U_seq,
                                    const Eigen::MatrixXd& X_seq_last,
                                    const Eigen::MatrixXd& U_seq_last,
                                    double* total_cost,
                                    double* total_state_cost,
                                    double* total_control_cost,
                                    double* total_auxiliary_cost) {
    double state_cost = 0.0, control_cost = 0.0, auxiliary_cost = 0.0;
    double total_control_jerk_cost = 0;
    double total_control_yaw_rate_dot_cost = 0;

    for (int i = 0; i < horizon_; i++) {
        state_cost = state_cost + getStateCost(X_seq.col(i + 1),
                                               X_seq_last.col(i + 1), i + 1);
        auxiliary_cost = auxiliary_cost +
                         getAuxiliaryCost(X_seq.col(i + 1), U_seq.col(i), i);
        if (i < horizon_ - 1) {
            control_cost = control_cost +
                           getControlCost(U_seq.col(i), U_seq_last.col(i + 1));
            total_control_jerk_cost +=
                U_seq.col(i)(0, 0) * U_seq.col(i)(0, 0) * control_cost_(0, 0);
            total_control_yaw_rate_dot_cost +=
                U_seq.col(i)(1, 0) * U_seq.col(i)(1, 0) * control_cost_(1, 1);
        } else {
            control_cost =
                control_cost + getControlCost(U_seq.col(i), U_seq.col(i));
            total_control_jerk_cost +=
                U_seq.col(i)(0, 0) * U_seq.col(i)(0, 0) * control_cost_(0, 0);
            total_control_yaw_rate_dot_cost +=
                U_seq.col(i)(1, 0) * U_seq.col(i)(1, 0) * control_cost_(1, 1);
        }
    }

    *total_cost = state_cost + control_cost + auxiliary_cost;
    *total_state_cost = state_cost;
    *total_control_cost = control_cost;
    *total_auxiliary_cost = auxiliary_cost;
}

void CilqrConstructor::findClosestLineSegmentOnSingleBoundary(
    const std::vector<BoundaryPoint>& boundary,
    const Eigen::MatrixXd& X,
    const int closest_idx,
    Eigen::MatrixXd* closest_segment_start,
    Eigen::MatrixXd* closest_segment_end,
    Eigen::MatrixXd* closest_segment_start_dir,
    Eigen::MatrixXd* closest_segment_end_dir) {

    Eigen::MatrixXd ego_pos(2, 1);
    ego_pos(0, 0) = X(0, 0);
    ego_pos(1, 0) = X(1, 0);

    // Get possible linear segments
    std::vector<std::pair<int, int>> segments;
    if (closest_idx == 0) {
        segments.emplace_back(0, 1);
    } else if (closest_idx == static_cast<int>(boundary.size()) - 1) {
        segments.emplace_back(closest_idx - 1, closest_idx);
    } else {
        segments.emplace_back(closest_idx - 1, closest_idx);
        segments.emplace_back(closest_idx, closest_idx + 1);
    }

    // Initialize characteristics of closest segment
    double closest_dist = std::numeric_limits<double>::infinity();
    std::pair<int, int> closest_idx_pair;
    Eigen::MatrixXd closest_projection(2, 1);
    double segment_ratio;

    // Find closest linear segment
    for (const auto& idx_pair : segments) {
        int i = idx_pair.first;
        int j = idx_pair.second;

        Eigen::MatrixXd p1(2, 1);
        p1 << boundary[i].pos.x, boundary[i].pos.y;

        Eigen::MatrixXd p2(2, 1);
        p2 << boundary[j].pos.x, boundary[j].pos.y;

        // Calculate Euclidean distance to line segment
        Eigen::MatrixXd projected_point =
            projectPointToSegment(ego_pos, p1, p2, &segment_ratio);

        double seg_dist = calculateEuclideanDistance(ego_pos, projected_point);

        // Update closest segments
        if (seg_dist < closest_dist) {
            closest_dist = seg_dist;
            *closest_segment_start = p1;
            *closest_segment_end = p2;
            closest_idx_pair = idx_pair;
            closest_projection = projected_point;

            // The projection point is on the right side of line segment
            if (segment_ratio > 1.0) {
                if (j + 1 >= boundary.size()) {
                    *closest_segment_end = closest_projection;
                    closest_idx_pair.second = j;
                } else {
                    (*closest_segment_end)(0, 0) = boundary[j + 1].pos.x;
                    (*closest_segment_end)(1, 0) = boundary[j + 1].pos.y;
                    closest_idx_pair.second = j + 1;
                }
            }

            // The projection point is on the left side of line segment
            else if (segment_ratio < 0.0) {  // NOLINT
                if (i - 1 < 0) {
                    *closest_segment_start = closest_projection;
                    closest_idx_pair.first = i;
                } else {
                    (*closest_segment_start)(0, 0) = boundary[i - 1].pos.x;
                    (*closest_segment_start)(1, 0) = boundary[i - 1].pos.y;
                    closest_idx_pair.first = i - 1;
                }
            }
        }
    }

    // Set closest segment index
    int start_idx = closest_idx_pair.first;
    int end_idx = closest_idx_pair.second;

    (*closest_segment_start_dir)(0, 0) = boundary[start_idx].dir.x;
    (*closest_segment_start_dir)(1, 0) = boundary[start_idx].dir.y;

    (*closest_segment_end_dir)(0, 0) = boundary[end_idx].dir.x;
    (*closest_segment_end_dir)(1, 0) = boundary[end_idx].dir.y;

    return;
}

void CilqrConstructor::findClosestLineSegmentOnBoundarySet(
    const std::vector<std::vector<BoundaryPoint>>& boundary,
    const Eigen::MatrixXd& X_seq,
    const std::string& axle_type) {

    // Clear previously stored closest boundary information
    if (axle_type == "rear") {
        rear_closest_boundary_.clear();
        rear_closest_boundary_direction_.clear();

        rear_closest_boundary_segment_.clear();
        rear_closest_boundary_segment_direction_.clear();
    } else if (axle_type == "front") {
        front_closest_boundary_.clear();
        front_closest_boundary_direction_.clear();

        front_closest_boundary_segment_.clear();
        front_closest_boundary_segment_direction_.clear();
    } else {
        return;
    }

    // Iterate over each boundary set
    // j ----- boundary index
    // i ----- trajectory point index
    // count - boundary point index
    for (int j = 0; j < boundary.size(); j++) {
        // Initialize matrices to store the closest boundary points
        // and their directions for each time step in the horizon.
        Eigen::MatrixXd closest_boundary =
            Eigen::MatrixXd::Zero(num_states_, horizon_);
        Eigen::MatrixXd closest_boundary_direction =
            Eigen::MatrixXd::Zero(static_cast<int>(3), horizon_);

        // Initialize matrices to store the closest line segment head and tail,
        // and their directions for each time step in the horizon.
        Eigen::MatrixXd closest_boundary_segment =
            Eigen::MatrixXd::Zero(static_cast<int>(4), horizon_);
        Eigen::MatrixXd closest_boundary_segment_direction =
            Eigen::MatrixXd::Zero(static_cast<int>(5), horizon_);

        // Start from the last known closest boundary index for this boundary
        // set
        int count = 0;
        if (axle_type == "rear") {
            count = rear_closest_boundary_start_index_[j];
        } else {
            count = front_closest_boundary_start_index_[j];
        }

        // Determine boundary direction
        BoundaryDirection boundary_type = boundary[j][0].boundary_type;

        // Iterate over each time step in the horizon
        for (int i = 0; i < horizon_; i++) {
            // If we have reached the end of the boundary set, use the last
            // boundary point. and last boundary segment.
            if (count > boundary[j].size() - 1) {
                count = boundary[j].size() - 1;
                closest_boundary(0, i) = boundary[j][count].pos.x;
                closest_boundary(1, i) = boundary[j][count].pos.y;
                closest_boundary_direction(0, i) = boundary[j][count].dir.x;
                closest_boundary_direction(1, i) = boundary[j][count].dir.y;

                closest_boundary_segment(0, i) = boundary[j][count - 1].pos.x;
                closest_boundary_segment(1, i) = boundary[j][count - 1].pos.y;
                closest_boundary_segment(2, i) = boundary[j][count].pos.x;
                closest_boundary_segment(3, i) = boundary[j][count].pos.y;

                closest_boundary_segment_direction(0, i) =
                    boundary[j][count - 1].dir.x;
                closest_boundary_segment_direction(1, i) =
                    boundary[j][count - 1].dir.y;
                closest_boundary_segment_direction(2, i) =
                    boundary[j][count].dir.x;
                closest_boundary_segment_direction(3, i) =
                    boundary[j][count].dir.y;
                closest_boundary_segment_direction(4, i) =
                    static_cast<int>(boundary_type);

                // Determine the side of the boundary relative to the trajectory
                // start point using a cross product.
                if (-boundary[j][count].dir.y *
                            (X_seq(0, 0) - boundary[j][count].pos.x) +
                        boundary[j][count].dir.x *
                            (X_seq(1, 0) - boundary[j][count].pos.y) <
                    0) {
                    closest_boundary_direction(2, i) = 1.0;
                    // closest_boundary_segment_direction(4, i) =  1.0;
                } else {
                    closest_boundary_direction(2, i) = -1.0;
                    // closest_boundary_segment_direction(4, i) = -1.0;
                }

                // Increment the count to continue to the next point in the next
                // iteration.
                count++;
                continue;
            }

            // Calculate the squared distance from the trajectory point to the
            // current and next boundary points.
            double cur_distance;
            cur_distance = pow(X_seq(0, i + 1) - boundary[j][count].pos.x, 2) +
                           pow(X_seq(1, i + 1) - boundary[j][count].pos.y, 2);

            double next_distance;
            next_distance =
                pow(X_seq(0, i + 1) - boundary[j][count + 1].pos.x, 2) +
                pow(X_seq(1, i + 1) - boundary[j][count + 1].pos.y, 2);

            // Iterate to find the closest boundary point to the current
            // trajectory point
            while ((next_distance < cur_distance) &&
                   ((count + 1) < boundary[j].size())) {
                count++;
                cur_distance = next_distance;
                next_distance =
                    pow(X_seq(0, i + 1) - boundary[j][count + 1].pos.x, 2) +
                    pow(X_seq(1, i + 1) - boundary[j][count + 1].pos.y, 2);
            }

            // Store the position of the found closest boundary point
            closest_boundary(0, i) = boundary[j][count].pos.x;
            closest_boundary(1, i) = boundary[j][count].pos.y;

            // Store the direction of the found closest boundary point.
            closest_boundary_direction(0, i) = boundary[j][count].dir.x;
            closest_boundary_direction(1, i) = boundary[j][count].dir.y;

            // Start finding the closest line segment
            Eigen::MatrixXd ego_pos = X_seq.block(0, i + 1, 2, 1);
            Eigen::MatrixXd closest_segment_start(2, 1);
            Eigen::MatrixXd closest_segment_end(2, 1);
            Eigen::MatrixXd closest_segment_start_dir(2, 1);
            Eigen::MatrixXd closest_segment_end_dir(2, 1);

            findClosestLineSegmentOnSingleBoundary(
                boundary[j], ego_pos, count, &closest_segment_start,
                &closest_segment_end, &closest_segment_start_dir,
                &closest_segment_end_dir);

            // Store the segment head and tail as well as directions
            closest_boundary_segment(0, i) = closest_segment_start(0, 0);
            closest_boundary_segment(1, i) = closest_segment_start(1, 0);
            closest_boundary_segment(2, i) = closest_segment_end(0, 0);
            closest_boundary_segment(3, i) = closest_segment_end(1, 0);

            closest_boundary_segment_direction(0, i) =
                closest_segment_start_dir(0, 0);
            closest_boundary_segment_direction(1, i) =
                closest_segment_start_dir(1, 0);
            closest_boundary_segment_direction(2, i) =
                closest_segment_end_dir(0, 0);
            closest_boundary_segment_direction(3, i) =
                closest_segment_end_dir(1, 0);
            closest_boundary_segment_direction(4, i) =
                static_cast<int>(boundary_type);

            // Store the lane type of the found closest boundary segment
            // and boundary point
            // Left = 1.0, Right = -1.0
            if (-boundary[j][count].dir.y *
                        (X_seq(0, i + 1) - boundary[j][count].pos.x) +
                    boundary[j][count].dir.x *
                        (X_seq(1, i + 1) - boundary[j][count].pos.y) <
                0) {
                closest_boundary_direction(2, i) = 1.0;
                // closest_boundary_segment_direction(4, i) = 1.0;
            } else {
                closest_boundary_direction(2, i) = -1.0;
                // closest_boundary_segment_direction(4, i) = -1.0;
            }
        }

        if (axle_type == "rear") {
            rear_closest_boundary_.emplace_back(closest_boundary);
            rear_closest_boundary_direction_.emplace_back(
                closest_boundary_direction);

            rear_closest_boundary_segment_.emplace_back(
                closest_boundary_segment);
            rear_closest_boundary_segment_direction_.emplace_back(
                closest_boundary_segment_direction);
        } else {
            front_closest_boundary_.emplace_back(closest_boundary);
            front_closest_boundary_direction_.emplace_back(
                closest_boundary_direction);

            front_closest_boundary_segment_.emplace_back(
                closest_boundary_segment);
            front_closest_boundary_segment_direction_.emplace_back(
                closest_boundary_segment_direction);
        }
    }
}

double CilqrConstructor::getSignedDistanceToBoundary(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& closest_boundary_segment,
    const Eigen::MatrixXd& closest_boundary_segment_direction,
    Eigen::MatrixXd* n_lambda,
    double* ratio_lambda) {
    if (closest_boundary_segment.rows() != 4 ||
        closest_boundary_segment.cols() != 1 ||
        closest_boundary_segment_direction.rows() != 5 ||
        closest_boundary_segment_direction.cols() != 1) {
        std::cerr << "Invalid boundary input";
        return boundary_safe_;
    }

    // Split 4*1 vector to two 2*1 vector
    Eigen::MatrixXd start_point = closest_boundary_segment.block(0, 0, 2, 1);
    Eigen::MatrixXd end_point = closest_boundary_segment.block(2, 0, 2, 1);
    Eigen::MatrixXd start_vector =
        closest_boundary_segment_direction.block(0, 0, 2, 1);
    Eigen::MatrixXd end_vector =
        closest_boundary_segment_direction.block(2, 0, 2, 1);

    // Get lane type / lane direction
    int temp_lane_type = closest_boundary_segment_direction(4, 0);
    BoundaryDirection lane_type =
        static_cast<BoundaryDirection>(temp_lane_type);

    // Get rotation matrix
    double l;
    Eigen::MatrixXd rotation_matrix =
        getRotationMatrixFromGlobalToLocal(start_point, end_point, &l);

    // Rotate points to local frame
    Eigen::MatrixXd ego_pos(2, 1);
    ego_pos(0, 0) = X(0, 0);
    ego_pos(1, 0) = X(1, 0);

    Eigen::MatrixXd shifted_ego_pos = ego_pos - start_point;

    Eigen::MatrixXd transformed_ego_pos = rotation_matrix * shifted_ego_pos;
    Eigen::MatrixXd vector_t1 = rotation_matrix * start_vector;
    Eigen::MatrixXd vector_t2 = rotation_matrix * end_vector;

    // Calculate signed distance
    double m1 = vector_t1(1, 0) / vector_t1(0, 0);
    double m2 = vector_t2(1, 0) / vector_t2(0, 0);
    double x = transformed_ego_pos(0, 0);
    double y = transformed_ego_pos(1, 0);

    *ratio_lambda = (m1 * y + x) / ((m1 - m2) * y + l);

    Eigen::MatrixXd local_n_lambda(2, 1);
    local_n_lambda(0, 0) = x - (*ratio_lambda) * l;
    local_n_lambda(1, 0) = y;

    Eigen::MatrixXd global_n_lambda =
        rotation_matrix.transpose() * local_n_lambda;

    *n_lambda = global_n_lambda;

    double dist = local_n_lambda.norm();
    double signed_dist = (y > 0) ? dist : -dist;

    if (lane_type == BoundaryDirection::LEFT) {
        signed_dist = -signed_dist;
    }

    return signed_dist;
}

double CilqrConstructor::getSignedDistanceBoundaryCost(
    const double q1,
    const double q2,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& closest_boundary_segment,
    const Eigen::MatrixXd& closest_boundary_segment_direction) {

    Eigen::MatrixXd ego_pos(2, 1);
    ego_pos(0, 0) = X(0, 0);
    ego_pos(1, 0) = X(1, 0);

    Eigen::MatrixXd n_lambda;
    double ratio_lambda;
    double signed_distance = getSignedDistanceToBoundary(
        ego_pos, closest_boundary_segment, closest_boundary_segment_direction,
        &n_lambda, &ratio_lambda);
    double c = boundary_safe_ - signed_distance;
    if (signed_distance > boundary_safe_) {
        c = 0;
    }
    double boundary_cost = q1 * exp(q2 * c);

    return boundary_cost;
}

Eigen::MatrixXd CilqrConstructor::getSignedDistanceBoundaryGrad(
    const double q1,
    const double q2,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& closest_boundary_segment,
    const Eigen::MatrixXd& closest_boundary_segment_direction,
    const std::string& axle_type) {

    Eigen::MatrixXd ego_pos(2, 1);
    ego_pos(0, 0) = X(0, 0);
    ego_pos(1, 0) = X(1, 0);

    Eigen::MatrixXd n_lambda;
    double ratio_lambda;
    double signed_distance = getSignedDistanceToBoundary(
        ego_pos, closest_boundary_segment, closest_boundary_segment_direction,
        &n_lambda, &ratio_lambda);

    double c = boundary_safe_ - signed_distance;


    Eigen::MatrixXd gradient_xy = n_lambda / (signed_distance + 1e-6);
    Eigen::MatrixXd gradient_c = Eigen::MatrixXd::Zero(num_states_, 1);

    gradient_c(0, 0) = -gradient_xy(0, 0);
    gradient_c(1, 0) = -gradient_xy(1, 0);

    Eigen::MatrixXd gradient = q1 * q2 * exp(q2 * c) * gradient_c;
    if (signed_distance > boundary_safe_) {
        gradient = gradient * 0;
    }
    if (axle_type == "front") {
        // Eigen::MatrixXd partial_front_to_state(num_states_, num_states_);

        // for (int i = 0; i < num_states_; i++) {
        //     partial_front_to_state(i, i) = 1;
        // }

        // const double L = wheelbase_;
        // const double theta = X(3, 0);
        // partial_front_to_state(0, 3) = -L * sin(theta);
        // partial_front_to_state(1, 3) = L * cos(theta);

        // return partial_front_to_state.transpose() * gradient;
        // return Eigen::MatrixXd::Zero(num_states_, 1);
    }

    return gradient;
}

Eigen::MatrixXd CilqrConstructor::getSignedDistanceBoundaryHessian(
    const double q1,
    const double q2,
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& closest_boundary_segment,
    const Eigen::MatrixXd& closest_boundary_segment_direction,
    const std::string& axle_type) {
    Eigen::MatrixXd grad = getSignedDistanceBoundaryGrad(
        q1_boundary_, q2_boundary_, X, closest_boundary_segment,
        closest_boundary_segment_direction, axle_type);

    Eigen::MatrixXd n_lambda;
    double ratio_lambda;
    double signed_distance = getSignedDistanceToBoundary(
        X, closest_boundary_segment, closest_boundary_segment_direction,
        &n_lambda, &ratio_lambda);

    double c = boundary_safe_ - signed_distance;

    Eigen::MatrixXd grad_origin = grad / (q1 * q2 * exp(q2 * c));
    Eigen::MatrixXd hessian =
        barrierFunctionHessian(q1_boundary_, q2_boundary_, 0, grad_origin);
    if (signed_distance > boundary_safe_) {
        hessian = hessian * 0;
    }
    return hessian;
}

