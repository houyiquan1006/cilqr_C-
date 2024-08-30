#include "cilqr_joint_planner/cilqr_joint_planner.hpp"
#include "acc_model/acc_model.hpp"
#include <iomanip> // For std::setw and std::left

Status CilqrJointPlanner::Init(CilqrConfig *cfg)
{
    cfg_ = *cfg;
    iters_ = 0;
    lambda_ = 1;
    lambda_factor_ = cfg_.ilqr.lambda_factor;
    max_lambda_ = cfg_.ilqr.max_lambda;
    wheelbase_ = cfg_.ilqr.wheel_base;
    update_dt_ = 1.0 / cfg_.ilqr.update_freq;
    dense_timestep_ = cfg_.ilqr.dense_timestep;
    sparse_timestep_ = cfg_.ilqr.sparse_timestep;
    horizon_ = cfg_.ilqr.horizon;
    dense_horizon_ = cfg_.ilqr.dense_horizon;
    tol_ = cfg_.ilqr.tol;
    max_iter_ = cfg_.ilqr.max_iter;
    min_iter_ = cfg_.ilqr.min_iter;
    num_states_ = cfg_.ilqr.num_states;
    num_ctrls_ = cfg_.ilqr.num_ctrls;
    num_auxiliary_ = cfg_.ilqr.num_auxiliary;

    x0_ = Eigen::MatrixXd::Zero(num_states_, 1);
    // u0_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    X_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    U_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    X_last_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    U_last_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    X_new_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    U_new_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    l_x_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    l_u_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    Q_x_ = Eigen::MatrixXd::Zero(num_states_, 1);
    Q_u_ = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    Q_xx_ = Eigen::MatrixXd::Zero(num_states_, num_states_);
    Q_ux_ = Eigen::MatrixXd::Zero(num_ctrls_, num_states_);
    Q_uu_ = Eigen::MatrixXd::Zero(num_ctrls_, num_ctrls_);
    k_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    K_ = std::vector<Eigen::MatrixXd>(horizon_);
    l_xx_ = std::vector<Eigen::MatrixXd>(horizon_ + 1);
    l_uu_ = std::vector<Eigen::MatrixXd>(horizon_);
    l_ux_ = std::vector<Eigen::MatrixXd>(horizon_);
    f_x_ = std::vector<Eigen::MatrixXd>(horizon_);
    f_u_ = std::vector<Eigen::MatrixXd>(horizon_);

    // vector_w_state_.clear();
    // vector_w_control_.clear();
    // vector_w_auxiliary_.clear();
    // std::copy(cfg_.cost_weight.w_states.begin(),
    //           cfg_.cost_weight.w_states.begin() + num_states_,
    //           std::back_inserter(vector_w_state_));
    // std::copy(cfg_.cost_weight.w_controls.begin(),
    //           cfg_.cost_weight.w_controls.begin() + num_ctrls_,
    //           std::back_inserter(vector_w_control_));
    // std::copy(cfg_.cost_weight.w_auxiliary.begin(),
    //           cfg_.cost_weight.w_auxiliary.begin() + num_auxiliary_,
    //           std::back_inserter(vector_w_auxiliary_));
    for (int i = 0; i < horizon_; i++)
    {
        l_xx_[i] = Eigen::MatrixXd::Zero(num_states_, num_states_);
        l_uu_[i] = Eigen::MatrixXd::Zero(num_ctrls_, num_ctrls_);
        l_ux_[i] = Eigen::MatrixXd::Zero(num_ctrls_, num_states_);
        f_x_[i] = Eigen::MatrixXd::Zero(num_states_, num_states_);
        f_u_[i] = Eigen::MatrixXd::Zero(num_states_, num_ctrls_);
        K_[i] = Eigen::MatrixXd::Zero(num_ctrls_, num_states_);
    }
    l_xx_[horizon_] = Eigen::MatrixXd::Zero(num_states_, num_states_);
    t_seq_.clear();
    for (uint32_t i = 0; i < cfg_.ilqr.dense_horizon + 1; ++i)
    {
        t_seq_.emplace_back(cfg_.ilqr.dense_timestep * i);
    }
    for (uint32_t i = 1; i < cfg_.ilqr.horizon - cfg_.ilqr.dense_horizon + 1; ++i)
    {
        t_seq_.emplace_back(cfg_.ilqr.dense_timestep * cfg_.ilqr.dense_horizon +
                            cfg_.ilqr.sparse_timestep * i);
    }
    // std::cout << "t_seq_: ";
    for (int i = 0; i < t_seq_.size(); i++)
    {
        // std::cout << t_seq_[i] << ", ";
    }
    std::cout << std::endl;
    return Status::SUCCESS;
}
Status CilqrJointPlanner::Update(PlanningFrame *frame)
{
    frame_ = frame;
    return Status::SUCCESS;
}

Status CilqrJointPlanner::setInitState()
{
    if (frame_ == nullptr)
    {
        std::cerr << "setInitState: frame_ is nullptr." << std::endl;
        return Status::NULL_PTR;
    }
    const TrajectoryPoint &start_point = frame_->start_point_;
    x0_(0, 0) = start_point.position.x;
    x0_(1, 0) = start_point.position.y;
    x0_(2, 0) = start_point.velocity;
    x0_(3, 0) = start_point.theta;
    x0_(4, 0) = start_point.acceleration;
    x0_(5, 0) = start_point.yaw_rate;

    return Status::SUCCESS;
}
void CilqrJointPlanner::findTargetTrajectory()
{
    for (int i = 0; i < horizon_ + 1; i++)
    {
        target_trajectory_(0, i) = frame_->ref_line_.traj_point_array[i].position.x;
        target_trajectory_(1, i) = frame_->ref_line_.traj_point_array[i].position.y;
        target_trajectory_(2, i) = frame_->ref_line_.traj_point_array[i].velocity;
        target_trajectory_(3, i) = frame_->ref_line_.traj_point_array[i].theta;
        target_trajectory_(4, i) = frame_->ref_line_.traj_point_array[i].acceleration;
        target_trajectory_(5, i) = frame_->ref_line_.traj_point_array[i].yaw_rate;
        // std::cout << "x: " << target_trajectory_(0, i) << ", " << "y: " << target_trajectory_(1, i) << std::endl;
        // std::cout << "v: " << target_trajectory_(2, i) << std::endl;
        // std::cout << "theta: " << target_trajectory_(3, i) << std::endl;
        // std::cout << "a: " << target_trajectory_(4, i) << std::endl;
        // std::cout << "yawrate: " << target_trajectory_(5, i) << std::endl;
    }
}
Status CilqrJointPlanner::findReferenceInfo()
{
    // target_trajectory_start_index_ = 0;
    target_trajectory_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    // target_last_trajectory_start_index_ = 0;

    // findTrajectoryStartIndex(local_plan_, x0_);
    // if (last_plan_.size() != 0) {
    //     findLastTrajectoryStartIndex(last_plan_, x0_);
    // }

    // // get start index of rear-cirlce boundary
    // cilqr_constructor_.findRearBoundaryStartIndex(boundary_, x0_);

    // Eigen::MatrixXd x0_front = Eigen::MatrixXd::Zero(num_states_, 1);
    // StateTransitfromReartoFront(&x0_front, x0_, 0, wheelbase_);

    // // get start index of front-cirlce boundary
    // cilqr_constructor_.findFrontBoundaryStartIndex(boundary_, x0_front);

    // // Set target trajectory using ACC forward simulation
    // const double ref_v =
    //     displaySpeedToActualSpeed(frame_->goal_point_.velocity);

    findTargetTrajectory();

    return Status::SUCCESS;
}

Status CilqrJointPlanner::setInitialSolution(
    const Eigen::MatrixXd &U_previous)
{

    X_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    U_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);

    X_.col(0) = x0_;
    for (int i = 0; i < horizon_ - 1; i++)
    {
        U_(0, i) = (U_previous(0, i + 1) - U_previous(0, i)) *
                       (update_dt_ / (t_seq_[i + 1] - t_seq_[i])) +
                   U_previous(0, i);
        U_(1, i) = (U_previous(1, i + 1) - U_previous(1, i)) *
                       (update_dt_ / (t_seq_[i + 1] - t_seq_[i])) +
                   U_previous(1, i);
    }
    U_(0, horizon_ - 1) = 0.0;
    U_(1, horizon_ - 1) = 0.0;
    for (int i = 0; i < horizon_; ++i)
    {
        std::cerr << U_previous(0, i) << "," << U_previous(1, i) << std::endl;
    }
    return Status::SUCCESS;
}

void CilqrJointPlanner::vehicleModelLinearization()
{
    for (int i = 0; i < horizon_; i++)
    {
        double ts = i < t_seq_.size() - 1 ? t_seq_[i + 1] - t_seq_[i]
                                          : sparse_timestep_;
        double velocity = X_(2, i);
        double theta = X_(3, i);
        double acceleration = X_(4, i);
        double yaw_rate = X_(5, i);
        double jerk = U_(0, i);
        double yaw_rate_dot = U_(1, i);
        f_x_[i](0, 0) = 1;
        f_x_[i](1, 1) = 1;
        f_x_[i](2, 2) = 1;
        f_x_[i](3, 3) = 1;
        f_x_[i](4, 4) = 1;
        f_x_[i](5, 5) = 1;
        f_x_[i](0, 2) = cos(theta) * ts;
        f_x_[i](0, 3) =
            -sin(theta) * (velocity * ts + acceleration * ts * ts / 2);
        f_x_[i](0, 4) = cos(theta) * ts * ts / 2;
        f_x_[i](1, 2) = sin(theta) * ts;
        f_x_[i](1, 3) =
            cos(theta) * (velocity * ts + acceleration * ts * ts / 2);
        f_x_[i](1, 4) = sin(theta) * ts * ts / 2;
        f_x_[i](2, 4) = ts;
        f_x_[i](3, 5) = ts;
        f_u_[i](4, 0) = ts;
        f_u_[i](5, 1) = ts;
    }
}

void CilqrJointPlanner::vehicleModel(const int i,
                                     Eigen::MatrixXd *X_ptr,
                                     Eigen::MatrixXd *U_ptr)
{
    Eigen::MatrixXd &X = *X_ptr;
    Eigen::MatrixXd &U = *U_ptr;
    double x = X(0, i);
    double y = X(1, i);
    double velocity = X(2, i);
    double theta = X(3, i);
    double acceleration = X(4, i);
    double yaw_rate = X(5, i);
    double jerk = U(0, i);
    double yaw_rate_dot = U(1, i);

    // yaw_rate = std::min(cfg_.constraint.control_constraints_max[1],
    //            std::max(cfg_.constraint.control_constraints_min[1], yaw_rate));
    // jerk = std::min(cfg_.constraint.control_constraints_max[0],
    //        std::max(cfg_.constraint.control_constraints_min[0], jerk));

    double ts =
        i < t_seq_.size() - 1 ? t_seq_[i + 1] - t_seq_[i] : sparse_timestep_;

    X(0, i + 1) = x + cos(theta) * (2 * velocity + acceleration * ts) * ts / 2;
    X(1, i + 1) = y + sin(theta) * (2 * velocity + acceleration * ts) * ts / 2;
    double v_next = velocity + acceleration * ts;
    // v_next = std::min(cfg_.constraint.state_constraints_max[2],
    //          std::max(cfg_.constraint.state_constraints_min[2], v_next));
    X(2, i + 1) = v_next;
    double theta_next = theta + yaw_rate * ts;
    // theta_next = std::min(cfg_.constraint.state_constraints_max[3],
    //         std::max(cfg_.constraint.state_constraints_min[3], theta_next));
    X(3, i + 1) = theta_next;
    double acceleration_next = acceleration + jerk * ts;
    // acceleration_next = std::min(cfg_.constraint.state_constraints_max[4],
    //                  std::max(cfg_.constraint.state_constraints_min[4],
    //                  acceleration_next));
    X(4, i + 1) = acceleration_next;
    double yaw_rate_next = yaw_rate + yaw_rate_dot * ts;
    X(5, i + 1) = yaw_rate_next;
    U(0, i) = jerk;
    U(1, i) = yaw_rate_dot;
}

// update U_new_ and X_new_ with alpha
Status CilqrJointPlanner::forwardPass(double alpha)
{
    X_new_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    U_new_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    X_new_.col(0) = x0_;
    for (int i = 0; i < horizon_; i++)
    {
        U_new_.col(i) =
            U_.col(i) + alpha * k_.col(i) + K_[i] * (X_new_.col(i) - X_.col(i));
        vehicleModel(i, &X_new_, &U_new_);
    }
    return Status::SUCCESS;
}

Status CilqrJointPlanner::forwardPass()
{
    X_new_ = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    U_new_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    X_new_.col(0) = x0_;
    for (int i = 0; i < horizon_; i++)
    {
        U_new_.col(i) =
            U_.col(i) + k_.col(i) + K_[i] * (X_new_.col(i) - X_.col(i));
        vehicleModel(i, &X_new_, &U_new_);
    }
    return Status::SUCCESS;
}

Status CilqrJointPlanner::backwardPass()
{
    // todo: cilqr_constraints
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    cilqr_constructor_.setTargetTrajectory(target_trajectory_);

    // cilqr_constructor_.findRearClosestBoundary(boundary_, X_);
    // cilqr_constructor_.findClosestLineSegmentOnBoundarySet(boundary_, X_,
    //                                                        "rear");

    // get front-circle state
    Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(num_states_, horizon_ + 1);
    for (int i = 0; i < horizon_ + 1; i++)
    {
        StateTransitfromReartoFront(&X_front, X_, i, wheelbase_);
    }
    // get front-circle boundary
    // cilqr_constructor_.findFrontClosestBoundary(boundary_, X_front);
    // cilqr_constructor_.findClosestLineSegmentOnBoundarySet(boundary_, X_front,
    //                                                        "front");

    k_ = Eigen::MatrixXd::Zero(num_ctrls_, horizon_);
    vehicleModelLinearization();
    Eigen::MatrixXd total_state_grad = Eigen::MatrixXd::Zero(num_states_, 1);
    Eigen::MatrixXd total_control_grad = Eigen::MatrixXd::Zero(num_ctrls_, 1);
    for (int i = 0; i < horizon_; i++)
    {
        Eigen::MatrixXd auxiliary_grad =
            Eigen::MatrixXd::Zero(num_states_ + num_ctrls_, 1); // dim = 8*1
        Eigen::MatrixXd auxiliary_hessian = Eigen::MatrixXd::Zero(
            num_states_ + num_ctrls_, num_states_ + num_ctrls_); // dim = 8*8
        auxiliary_grad = cilqr_constructor_.getAuxiliaryCostGrad(X_.col(i + 1),
                                                                 U_.col(i), i);
        auxiliary_hessian = cilqr_constructor_.getAuxiliaryCostHessian(
            X_.col(i + 1), U_.col(i), i);

        if (i < horizon_ - 1)
        {
            // i th state in X_ should correspond to i+1 th state in X_last
            l_x_.col(i + 1) =
                cilqr_constructor_.getStateCostGrad(X_.col(i + 1),
                                                    X_last_.col(i + 1), i + 1) +
                auxiliary_grad.block(0, 0, num_states_, 1); // dim = 6*1
            l_u_.col(i) = cilqr_constructor_.getControlCostGrad(
                              U_.col(i), U_last_.col(i + 1)) +
                          auxiliary_grad.block(num_states_, 0, num_ctrls_,
                                               1); // dim = 2*1
            for (int ii(0); ii < num_states_; ii++)
            {
                total_state_grad(ii, 0) += fabs(l_x_(ii, i + 1));
            }
            for (int ii(0); ii < num_ctrls_; ii++)
            {
                total_state_grad(ii, 0) += fabs(l_x_(ii, i + 1));
            }
        }
        else
        {
            // the last state in X_ don't use regularization cost
            l_x_.col(i + 1) = cilqr_constructor_.getStateCostGrad(
                                  X_.col(i + 1), X_.col(i + 1), i + 1) +
                              auxiliary_grad.block(0, 0, num_states_, 1);
            l_u_.col(i) =
                cilqr_constructor_.getControlCostGrad(U_.col(i), U_.col(i)) +
                auxiliary_grad.block(num_states_, 0, num_ctrls_, 1);
            for (int ii(0); ii < num_states_; ii++)
            {
                total_state_grad(ii, 0) += fabs(l_x_(ii, i + 1));
            }
            for (int ii(0); ii < num_ctrls_; ii++)
            {
                total_state_grad(ii, 0) += fabs(l_x_(ii, i + 1));
            }
        }
        l_xx_[i + 1] =
            cilqr_constructor_.getStateCostHessian(X_.col(i + 1), i + 1) +
            auxiliary_hessian.block(0, 0, num_states_,
                                    num_states_); // dim = 6*6
        l_uu_[i] = cilqr_constructor_.getControlCostHessian(U_.col(i)) +
                   auxiliary_hessian.block(num_states_, num_states_, num_ctrls_,
                                           num_ctrls_); // dim = 2*2
        l_ux_[i] = auxiliary_hessian.block(num_states_, 0, num_ctrls_,
                                           num_states_); // dim = 2*6
        K_[i] = Eigen::MatrixXd::Zero(num_ctrls_, num_states_);
    }

    Eigen::MatrixXd V_x = l_x_.col(horizon_);
    Eigen::MatrixXd V_xx = l_xx_[horizon_];
    for (int i = horizon_ - 1; i >= 0; i--)
    {
        Q_x_ = l_x_.col(i) + f_x_[i].transpose() * V_x;
        Q_u_ = l_u_.col(i) + f_u_[i].transpose() * V_x;
        Q_xx_ = l_xx_[i] + f_x_[i].transpose() * V_xx * f_x_[i];
        Q_uu_ = l_uu_[i] + f_u_[i].transpose() * V_xx * f_u_[i];
        Q_ux_ = l_ux_[i] + f_u_[i].transpose() * V_xx * f_x_[i]; // + l_ux_[i]
        if (Q_uu_(0, 0) >= 1e100)
            Q_uu_(0, 0) = 1e100;
        if (Q_uu_(0, 0) <= -1e100)
            Q_uu_(0, 0) = -1e100;
        try
        {
            if (Q_uu_.array().isNaN().any())
            {
                throw std::runtime_error("NaN in Q_uu_");
                std::cout << "NaN in Q_uu_" << std::endl;
            }
        }
        catch (const std::runtime_error &e)
        {
            std::cout << e.what() << std::endl;
            return Status::INTERNAL_ERROR;
        }
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_);
        Eigen::MatrixXd Values = es.pseudoEigenvalueMatrix();
        Eigen::MatrixXd Vectors = es.pseudoEigenvectors();
        for (int j = 0; j < Values.rows(); j++)
        {
            if (Values(j, j) < 0)
            {
                Values(j, j) = 0;
            }
            Values(j, j) += lambda_;
            Values(j, j) = 1.0 / Values(j, j);
        }
        Eigen::MatrixXd Q_uu_inv;
        Q_uu_inv = Vectors * Values * Vectors.transpose();
        k_.col(i) = -Q_uu_inv * Q_u_;
        K_[i] = -Q_uu_inv * Q_ux_;
        V_x = Q_x_ - K_[i].transpose() * Q_uu_ * k_.col(i);
        V_xx = Q_xx_ - K_[i].transpose() * Q_uu_ * K_[i];
        for (int i = 0; i < V_xx.rows(); ++i)
        {
            for (int j = 0; j < V_xx.cols(); ++j)
            {
                if (V_xx(i, j) >= 1e100)
                    V_xx(i, j) = 1e100;
                if (V_xx(i, j) <= -1e100)
                    V_xx(i, j) = -1e100;
            }
        }
    }
    return Status::SUCCESS;
}

Status CilqrJointPlanner::getOptimalControlSeq()
{
    // Overwrite state and control sequence with
    // pre-calculated result.
    for (int i = 0; i < horizon_; i++)
    {
        vehicleModel(i, &X_, &U_);
    }

    // Initiallize optimizer utils.
    // target_last_trajectory_ = Eigen::MatrixXd::Zero(num_states_, horizon_);

    lambda_ = 1;
    double J_new = 0.0, J_new_state = 0.0, J_new_control = 0.0,
           J_new_auxiliary = 0.0;
    double J_old = 0.0, J_old_state = 0.0, J_old_control = 0.0,
           J_old_auxiliary = 0.0;
    for (int i = 0; i < max_iter_; i++)
    {
        if (backwardPass() != Status::SUCCESS)
        {
            return Status::INTERNAL_ERROR;
        }
        // line search
        bool accepted = false;
        cilqr_constructor_.getTotalCost(X_, U_, X_last_, U_last_, &J_old,
                                        &J_old_state, &J_old_control,
                                        &J_old_auxiliary);
        double old_tra_cost, old_x_cost, old_y_cost;
        std::tie(old_tra_cost, old_x_cost, old_y_cost) = cilqr_constructor_.getTotalTraCost(X_);
        double old_boundary_cost = cilqr_constructor_.getTotalBoundaryCost(X_);
        double old_state_barrier_cost =
            cilqr_constructor_.getTotalStateBarrierCost(X_);
        for (auto alpha : alphas_)
        {
            forwardPass(alpha);
            cilqr_constructor_.getTotalCost(X_new_, U_new_, X_last_, U_last_,
                                            &J_new, &J_new_state,
                                            &J_new_control, &J_new_auxiliary);
            double new_tra_cost, new_x_cost, new_y_cost;
            std::tie(new_tra_cost, new_x_cost, new_y_cost) = cilqr_constructor_.getTotalTraCost(X_new_);
            double new_boundary_cost =
                cilqr_constructor_.getTotalBoundaryCost(X_new_);
            double new_state_barrier_cost =
                cilqr_constructor_.getTotalStateBarrierCost(X_new_);

            std::cout << "iter: " << i << ", lambda = " << lambda_
                      << " ................\n";
            std::cout << "J_old = " << J_old << ", J_old_state = " << J_old_state
                      << ", J_old_control = " << J_old_control
                      << ", J_old_auxiliary = " << J_old_auxiliary << ", old_tra_cost = "
                      << old_tra_cost << " ( " << old_x_cost << ", " << old_y_cost << " ) "
                      << ", old_boundary_cost = " << old_boundary_cost << ", old_state_barrier_cost = "
                      << old_state_barrier_cost << "\n";
            std::cout << "J_new = " << J_new << ", J_new_state = " << J_new_state
                      << ", J_new_control = " << J_new_control
                      << ", J_new_auxiliary = " << J_new_auxiliary << ", new_tra_cost = "
                      << new_tra_cost << " ( " << new_x_cost << ", " << new_y_cost << " ) "
                      << ", new_boundary_cost = " << new_boundary_cost << ", new_state_barrier_cost = "
                      << new_state_barrier_cost << "\n";

            if (J_new < J_old)
            {
                X_ = X_new_;
                U_ = U_new_;
                lambda_ = lambda_ / lambda_factor_;

                lambda_ = std::max(lambda_, 1e-6);
                if (fabs(J_new - J_old) < tol_ * J_old && i >= min_iter_)
                {
                    // X_last_ = X_;
                    // U_last_ = U_;
                    std::cout << "convergence success" << std::endl;
                    std::cout << "iter: " << iters_ << " lambda: " << lambda_
                              << " Jnew:" << J_new << " Jold: " << J_old << std::endl;
                    return Status::SUCCESS;
                }
                J_old = J_new;
                accepted = true;
                break;
            }
        }
        if (!accepted)
        {
            lambda_ = lambda_ * lambda_factor_;
            if (lambda_ > max_lambda_)
            {
                // X_last_ = X_;
                // U_last_ = U_;
                std::cout << "lambda_ > max_lambda_." << " convergence failed" << std::endl;
                if (std::isnan(J_old))
                {
                    std::cout << "CILQR result is nan!" << std::endl;
                    return Status::INTERNAL_ERROR;
                }
                return Status::SUCCESS;
            }
        }
        iters_ += 1;
    }

    std::cout << "Iteration exceeded the maximum limit!" << std::endl;
    if (std::isnan(J_old))
    {
        std::cout << "CILQR result is nan!" << std::endl;
        return Status::INTERNAL_ERROR;
    }
    return Status::SUCCESS;
}
Status CilqrJointPlanner::RunCilqrPlanner(Trajectory *out_path)
{

    // // set last plan
    // if (frame_->egopos_vec_.size() != 0) {
    //     if (setLastPlan() != Status::SUCCESS) {
    //         std::cout << "setLastPlan is failed.";
    //         return Status::INTERNAL_ERROR;
    //     }
    // }

    // set refline
    // if (setLocalPlan() != Status::SUCCESS) {
    //     std::cout << "setLocalPlan is failed." << std::endl;
    //     return Status::INTERNAL_ERROR;
    // }

    // // set boundary
    // if (setBoundary() != Status::SUCCESS) {
    //     std::cout << "setBoundary is failed.";
    //     return Status::INTERNAL_ERROR;
    // }

    // set init_state
    if (setInitState() != Status::SUCCESS)
    {
        std::cerr << "setInitState is failed." << std::endl;
        return Status::INTERNAL_ERROR;
    }

    // chooseTargetObjectAndTrajectory(frame_->model_obstacles_);
    if (cilqr_constructor_.Init(
            &cfg_, frame_->model_obstacles_) != Status::SUCCESS)
    {
        std::cerr << "cilqr_constructor_.Init is failed." << std::endl;
        return Status::INTERNAL_ERROR;
    }

    if (findReferenceInfo() != Status::SUCCESS)
    {
        std::cerr << "Set reference info failed." << std::endl;
        return Status::INTERNAL_ERROR;
    }
    if (setInitialSolution(U_last_) !=
        Status::SUCCESS)
    {
        std::cerr << "setInitialSolution failed." << std::endl;
        return Status::INTERNAL_ERROR;
    }
    if (getOptimalControlSeq() != Status::SUCCESS)
    {
        // setFallbackSolutionfromLastSolution();
        // extractOutputTrajectory(out_path);
        X_last_ = X_;
        U_last_ = U_;
        std::cout << "CILQR computation failed" << std::endl;
        return Status::SUCCESS;
    }
    extractOutputTrajectory(out_path);
    checkTrajectoryStatus(out_path);
    return Status::SUCCESS;
}

Status CilqrJointPlanner::extractOutputTrajectory(
    Trajectory *out_path)
{
    // extract trajectory
    std::vector<TrajectoryPoint> &trajectory =
        out_path->traj_point_array;
    std::vector<cv::Point2f> pos_x_y;
    for (int i = 0; i < horizon_ + 1; i++)
    {
        cv::Point2f pt;
        pt.x = X_(0, i);
        pt.y = X_(1, i);
        pos_x_y.push_back(pt);
    }

    int n = static_cast<int>(pos_x_y.size());
    if (n <= 0)
        return Status::INTERNAL_ERROR;
    trajectory.resize(n);
    // std::cout << "pos_x_y.size: " << static_cast<int>(pos_x_y.size()) << " n: " << n << std::endl;

    // start
    TrajectoryPoint &start = trajectory.front();
    start.position = pos_x_y[0];
    start.direction = frame_->start_point_.direction;
    start.theta = frame_->start_point_.theta;
    // start.curvature = frame_->start_point_.curvature;
    start.velocity = frame_->start_point_.velocity;
    start.acceleration = frame_->start_point_.acceleration;
    start.yaw_rate = frame_->start_point_.yaw_rate;
    start.jerk = frame_->start_point_.jerk;
    start.yaw_rate_dot = frame_->start_point_.yaw_rate_dot;
    start.sum_distance = 0.0;
    start.time_difference = t_seq_[0];
    start.timestamp = out_path->traj_base_timestamp_ns;
    // std::cout << std::left << std::setw(25) << "CILQR trajectory i"
    //           << std::setw(10) << "x"
    //           << std::setw(15) << "y"
    //           << std::setw(15) << "v"
    //           << std::setw(15) << "theta"
    //           << std::setw(15) << "acceleration"
    //           << std::setw(15) << "yaw rate"
    //           << std::setw(15) << "jerk"
    //           << std::setw(15) << "yawratedot"
    //           << std::endl;
    // std::cout << std::left << std::setw(25) << ("CILQR trajectory " + std::to_string(0) + ": ")
    //           << std::setw(10) << start.position.x
    //           << std::setw(15) << start.position.y
    //           << std::setw(15) << start.velocity
    //           << std::setw(15) << start.theta
    //           << std::setw(15) << start.acceleration
    //           << std::setw(15) << start.yaw_rate
    //           << std::setw(15) << start.jerk
    //           << std::setw(15) << start.yaw_rate_dot
    //           << std::endl;

    // intermediate points
    for (int i = 1; i < n; ++i)
    {
        TrajectoryPoint &point = trajectory[i];

        point.position = pos_x_y[i];
        cv::Point2f raw_direction = pos_x_y[i + 1] - pos_x_y[i - 1];
        point.direction =
            raw_direction * (1 / (cv::norm(raw_direction) + 10e-8));

        point.velocity = X_(2, i);
        double heading = X_(3, i);
        normalizeHeading(&heading);
        point.theta = heading;
        // point.curvature = X_(3, i);
        // point.kapparate = U_(0, i);
        point.acceleration = X_(4, i);
        point.yaw_rate = X_(5, i);
        if (i < n - 1)
        {
            point.jerk = U_(0, i);
            point.yaw_rate_dot = U_(1, i);
        }

        point.sum_distance = trajectory[i - 1].sum_distance +
                             cv::norm(pos_x_y[i] - pos_x_y[i - 1]);
        point.time_difference = t_seq_[i];
        point.timestamp =
            out_path->traj_base_timestamp_ns + i * static_cast<int>(1e8);

        std::cout << std::left << std::setw(25) << ("CILQR trajectory " + std::to_string(i) + ": ")
                  << std::setw(10) << point.position.x
                  << std::setw(15) << point.position.y
                  << std::setw(15) << point.velocity
                  << std::setw(15) << point.theta
                  << std::setw(15) << point.acceleration
                  << std::setw(15) << point.yaw_rate
                  << std::setw(15) << point.jerk
                  << std::setw(15) << point.yaw_rate_dot
                  << std::endl;
    }
    // visualizeTrajectory(trajectory);
    return Status::SUCCESS;
}

Status CilqrJointPlanner::checkTrajectoryStatus(Trajectory *out_path)
{
    if (!isTrajectoryFeasible(*out_path))
    {
        // TODO(ZBW): fallback solution is yet to define
        setFallbackSolutionfromLastSolution();
        extractOutputTrajectory(out_path);
        // Note: is_valid is always set to true now but keep the signal for
        // later use. if set to false, longilatplanner uses history trajectory.
        std::cout << "Optimized Trajectory is not feasible, Use last successfully "
                     "optimized control sequence to interpolate fallback solution "
                     "as well as initialization of next iteration "
                  << std::endl;
    }
    X_last_ = X_;
    U_last_ = U_;
    return Status::SUCCESS;
}
void CilqrJointPlanner::normalizeHeading(double *heading)
{
    double a = std::fmod(*heading + M_PI, 2.0 * M_PI);
    if (a < 0.0)
    {
        a += (2.0 * M_PI);
    }
    *heading = a - M_PI;
}

bool CilqrJointPlanner::isTrajectoryFeasible(
    const Trajectory &out_path)
{
    int feasibility_check_horizon =
        std::min(cfg_.ilqr.feasibility_check_horizon,
                 static_cast<int>(out_path.traj_point_array.size()));
    const std::vector<TrajectoryPoint> &ego_pts = std::vector<TrajectoryPoint>(
        out_path.traj_point_array.begin(),
        out_path.traj_point_array.begin() + feasibility_check_horizon);
    if (ego_pts.empty())
    {
        std::cout << "ego trajectory empty." << std::endl;
        return false;
    }
    // TODO(ZBW) read from vehicle param
    double ego_length = 5.0;
    double ego_radius = std::sqrt(std::pow(ego_length / 4, 2) +
                                  std::pow(cfg_.ego_vehicle.width / 2, 2));
    std::vector<cv::Point2f> ego_front_circle_trajectory;
    std::vector<cv::Point2f> ego_rear_circle_trajectory;

    for (auto pt : ego_pts)
    {
        // ego position is at the rear coordinate of vehicle
        ego_front_circle_trajectory.emplace_back(
            cv::Point2f(pt.position.x + std::cos(pt.theta) *
                                            (ego_length / 4 + wheelbase_ / 2),
                        pt.position.y + std::sin(pt.theta) *
                                            (ego_length / 4 + wheelbase_ / 2)));
        ego_rear_circle_trajectory.emplace_back(
            cv::Point2f(pt.position.x - std::cos(pt.theta) *
                                            (ego_length / 4 - wheelbase_ / 2),
                        pt.position.y - std::sin(pt.theta) *
                                            (ego_length / 4 - wheelbase_ / 2)));
    }
    // check model obstacles collision

    for (const auto &obs : frame_->model_obstacles_)
    {
        if (obs.trajectory_array.empty())
        {
            // std::cout << " The obstacles " << obs.id << " traj points is empty." << std::endl;
            continue;
        }
        for (PredictionTrajectory traj : obs.trajectory_array)
        {
            // assume there are only 2 trajectory for each obstacle and we
            // choose the one with higher probability
            if (traj.score < 0.5)
            {
                continue;
            }
            double obs_radius = std::sqrt(std::pow(obs.length / 4, 2) +
                                          std::pow(obs.width / 2, 2));
            std::vector<cv::Point2f> obs_front_circle_trajectory;
            std::vector<cv::Point2f> obs_rear_circle_trajectory;
            const std::vector<PredictionTrajectoryPoint> &obs_pts =
                traj.trajectory_point_array;
            if (obs_pts.size() == 0)
            {
                continue;
            }
            // TODO(ZBW): make sure the time diff of each point of ego_pts and
            // obs_pts are the same
            for (auto pt : obs_pts)
            {
                // TODO(ZBW): check the definition of direction
                obs_front_circle_trajectory.emplace_back(cv::Point2f(
                    pt.position.x + pt.direction.x * obs.length / 4,
                    pt.position.y + pt.direction.y * obs.length / 4));
                obs_rear_circle_trajectory.emplace_back(cv::Point2f(
                    pt.position.x - pt.direction.x * obs.length / 4,
                    pt.position.y - pt.direction.y * obs.length / 4));
            }
            int common_size = std::min(ego_pts.size(), obs_pts.size());
            cv::Point2f ego_front_to_obs_front, ego_front_to_obs_rear,
                ego_rear_to_obs_front, ego_rear_to_obs_rear;
            for (int i = 0; i < common_size; i++)
            {
                ego_front_to_obs_front = ego_front_circle_trajectory[i] -
                                         obs_front_circle_trajectory[i];
                ego_front_to_obs_rear = ego_front_circle_trajectory[i] -
                                        obs_front_circle_trajectory[i];
                ego_rear_to_obs_front = ego_rear_circle_trajectory[i] -
                                        obs_front_circle_trajectory[i];
                ego_rear_to_obs_rear = ego_rear_circle_trajectory[i] -
                                       obs_front_circle_trajectory[i];
                if (cv::norm(ego_front_to_obs_front) <
                    (ego_radius + obs_radius))
                {
                    std::cout << "trajectory obstacle front_to_front collision check failed" << std::endl;
                    return false;
                }
                if (cv::norm(ego_front_to_obs_rear) <
                    (ego_radius + obs_radius))
                {
                    std::cout << "trajectory obstacle front_to_rear collision check failed" << std::endl;
                    return false;
                }
                if (cv::norm(ego_rear_to_obs_front) <
                    (ego_radius + obs_radius))
                {
                    std::cout << "trajectory obstacle rear_to_front collision check failed" << std::endl;
                    return false;
                }
                if (cv::norm(ego_rear_to_obs_rear) <
                    (ego_radius + obs_radius))
                {
                    std::cout << "trajectory obstacle rear_to_rear collision check failed" << std::endl;
                    return false;
                }
            }
        }
    }
    // // check distance to boundary obstacles
    // std::vector<Eigen::MatrixXd> rear_closest_boundary_segments =
    //     cilqr_constructor_.getRearClosestBoundarySegments();

    // std::vector<Eigen::MatrixXd> rear_closest_boundary_segment_directions =
    //     cilqr_constructor_.getRearClosestBoundarySegmentDirections();

    // std::vector<Eigen::MatrixXd> front_closest_boundary_segments =
    //     cilqr_constructor_.getFrontClosestBoundarySegments();

    // std::vector<Eigen::MatrixXd> front_closest_boundary_segment_directions =
    //     cilqr_constructor_.getFrontClosestBoundarySegmentDirections();

    // // Traverse feasibility check horizon
    // for (int i = 1; i < feasibility_check_horizon; i++) {
    //     // Traverse left and right boundary
    //     for (int j = 0; j < rear_closest_boundary_segments.size(); j++) {
    //         Eigen::MatrixXd X(2, 1);
    //         X << ego_pts[i].position.x, ego_pts[i].position.y;

    //         Eigen::MatrixXd rear_segment =
    //             rear_closest_boundary_segments[j].col(i);
    //         Eigen::MatrixXd rear_segment_direction =
    //             rear_closest_boundary_segment_directions[j].col(i);

    //         Eigen::MatrixXd n_lambda;
    //         double ratio_lambda;
    //         const double signed_distance =
    //             cilqr_constructor_.getSignedDistanceToBoundary(
    //                 X, rear_segment, rear_segment_direction, &n_lambda,
    //                 &ratio_lambda);

    //         if (cfg_.constraint.feasibility_check_boundary_safe -
    //                 signed_distance >
    //             0) {
    //             AD_LERROR(TrajectoryFeasibility)
    //                 << "Trajectory rear boundary safe distance check failed at "
    //                 << i << "th step. Signed distance = " << signed_distance;

    //             return false;
    //         }
    //     }

    //     // Traverse left and right boundary
    //     for (int j = 0; j < front_closest_boundary_segments.size(); j++) {
    //         Eigen::MatrixXd X(2, 1);
    //         X << ego_pts[i].position.x, ego_pts[i].position.y;

    //         Eigen::MatrixXd X_front = Eigen::MatrixXd::Zero(2, 1);
    //         X_front(0, 0) = X(0, 0) + ego_pts[i].direction.x * wheelbase_;
    //         X_front(1, 0) = X(1, 0) + ego_pts[i].direction.y * wheelbase_;

    //         Eigen::MatrixXd front_segment =
    //             front_closest_boundary_segments[j].col(i);
    //         Eigen::MatrixXd front_segment_direction =
    //             front_closest_boundary_segment_directions[j].col(i);

    //         Eigen::MatrixXd n_lambda;
    //         double ratio_lambda;
    //         const double signed_distance =
    //             cilqr_constructor_.getSignedDistanceToBoundary(
    //                 X_front, front_segment, front_segment_direction, &n_lambda,
    //                 &ratio_lambda);

    //         if (cfg_.constraint.feasibility_check_boundary_safe -
    //                 signed_distance >
    //             0) {
    //             AD_LERROR(TrajectoryFeasibility)
    //                 << "Trajectory front boundary safe distance check failed "
    //                    "at "
    //                 << i << "th step. Signed distance = " << signed_distance;

    //             return false;
    //         }
    //     }
    // }

    // check state limit and control limit
    for (auto pt : ego_pts)
    {
        if (pt.velocity >
            cfg_.constraint.feasibility_check_state_constraints_max[2])
        {
            std::cout << "trajectory velocity max limit check failed" << std::endl;
            std::cout << "ego vel: " << pt.velocity << "max vel: "
                      << cfg_.constraint.feasibility_check_state_constraints_max[2] << std::endl;
            return false;
        }
        if (pt.velocity <
            cfg_.constraint.feasibility_check_state_constraints_min[2])
        {
            std::cout << "trajectory velocity min limit check failed" << std::endl;
            std::cout << "ego vel: " << pt.velocity << "max vel: "
                      << cfg_.constraint.feasibility_check_state_constraints_min[2] << std::endl;
            return false;
        }
        // if (pt.theta >
        //     cfg_.constraint.feasibility_check_state_constraints_max[3]) {
        //     std::cout << "trajectory theta max limit check failed" << std::endl;
        //     std::cout << "ego theta: " << pt.theta << "max theta: "
        //                     << cfg_.constraint.feasibility_check_state_constraints_max[3] << std::endl;
        //     return false;
        // }
        // if (pt.theta <
        //     cfg_.constraint.feasibility_check_state_constraints_min[3]) {
        //     std::cout << "trajectory theta min limit check failed" << std::endl;
        //     std::cout << "ego theta: " << pt.theta << "min theta: "
        //         << cfg_.constraint.feasibility_check_state_constraints_min[3] << std::endl;
        //     return false;
        // }
        // if (pt.acceleration >
        // cfg_.constraint.feasibility_check_state_constraints_max[4]) {
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "trajectory acceleration max limit check failed";
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "ego acceleration: " << pt.acceleration
        //         << "max acceleration: " <<
        //         cfg_.constraint.feasibility_check_state_constraints_max[4];
        //     return false;
        // }
        // if (pt.acceleration <
        // cfg_.constraint.feasibility_check_state_constraints_min[4]) {
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "trajectory acceleration min limit check failed";
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "ego acceleration: " << pt.acceleration
        //         << "min acceleration: "
        //         <<
        //         cfg_.constraint.feasibility_check_state_constraints_min[4];
        //     return false;
        // }
        // if (pt.yaw_rate >
        // cfg_.constraint.feasibility_check_state_constraints_max[5]) {
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "trajectory yaw_rate max limit check failed";
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "ego yaw_rate: " << pt.yaw_rate << "max yaw_rate: "
        //         <<
        //         cfg_.constraint.feasibility_check_state_constraints_max[5];
        //     return false;
        // }
        // if (pt.yaw_rate <
        // cfg_.constraint.feasibility_check_state_constraints_min[5]) {
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "trajectory yaw_rate min limit check failed";
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "ego yaw_rate: " << pt.yaw_rate << "min yaw_rate: "
        //         <<
        //         cfg_.constraint.feasibility_check_state_constraints_min[5];
        //     return false;
        // }
        // if (pt.yaw_rate_dot >
        // cfg_.constraint.feasibility_check_control_constraints_max[1]) {
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "trajectory yaw_rate_dot max limit check failed";
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "ego yaw_rate_dot: " << pt.yaw_rate_dot
        //         << "max yaw_rate_dot: " <<
        //         cfg_.constraint.feasibility_check_control_constraints_max[1];
        //     return false;
        // }
        // if (pt.yaw_rate_dot <
        // cfg_.constraint.feasibility_check_control_constraints_min[1]) {
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "trajectory yaw_rate_dot min limit check failed";
        //     AD_LERROR(TrajectoryFeasibility)
        //         << "ego yaw_rate_dot: " << pt.jerk
        //         << "min yaw_rate_dot: " <<
        //         cfg_.constraint.feasibility_check_control_constraints_min[1];
        //     return false;
        // }
    }
    return true;
}
void CilqrJointPlanner::setFallbackSolutionfromLastSolution()
{
    // TODO(ZBW): adapt fallback solution later
    return;
}
void CilqrJointPlanner::visualizeTrajectory(const std::vector<TrajectoryPoint> &trajectory)
{
    // // 1. display trajectory
    // cv::Mat grid_map_disp(500, 500, CV_8UC1, cv::Scalar(0));
    // cv::threshold(grid_map_disp, grid_map_disp, 75, 255, cv::THRESH_BINARY);
    // // for (int i = 0; i + 1 < trajectory.size(); ++i) {
    // //     cv::Point2f grid_pt_0 = trajectory[i].position * 20.0;
    // //     cv::Point2f grid_pt_1 = trajectory[i + 1].position * 20.0;
    // //     // cv::circle(grid_map_disp, grid_pt_0, 1, cv::Scalar(0,
    // //     // 255, 0),
    // //     //            CV_FILLED);
    // //     cv::line(grid_map_disp, grid_pt_0, grid_pt_1, cv::Scalar(255, 0, 0), 3);
    // // }

    // // 2. display start and goal point
    // cv::Point2f grid_pt_begin = frame_->start_point_.position * 20.0;
    // cv::circle(grid_map_disp, grid_pt_begin, 16, cv::Scalar(203, 192, 255),
    //            CV_FILLED);
    // cv::Point2f grid_pt_end = trajectory.back().position * 20.0;
    // cv::circle(grid_map_disp, grid_pt_end, 2, cv::Scalar(203, 192, 255),
    //            CV_FILLED);

    // // // 3. display polygon obstacle
    // // for (int i = 0; i < frame_->model_obstacles_.size(); i++) {
    // //     const std::vector<cv::Point2f>& obs_polygon_contour =
    // //         frame_->model_obstacles_[i].polygon_contour;

    // //     cv::Point2f grid_pt_start_0 = obs_polygon_contour.front() * 20.0;
    // //     cv::Point2f grid_pt_end_0 = obs_polygon_contour.back() * 20.0;
    // //     for (int j = 0; j + 1 < obs_polygon_contour.size(); ++j) {
    // //         cv::Point2f grid_pt_0 = obs_polygon_contour[j] * 20.0;
    // //         cv::Point2f grid_pt_1 = obs_polygon_contour[j + 1] * 20.0;
    // //         cv::circle(grid_map_disp, grid_pt_0, 2, cv::Scalar(0, 0, 255),
    // //                    CV_FILLED);
    // //         cv::line(grid_map_disp, grid_pt_0, grid_pt_1, cv::Scalar(0, 0, 255),
    // //                  2);
    // //         if (j + 2 == obs_polygon_contour.size()) {
    // //             cv::line(grid_map_disp, grid_pt_end_0, grid_pt_start_0,
    // //                      cv::Scalar(0, 0, 255), 2);
    // //         }
    // //     }

    // //     // prediction trajectory point of obstacle
    // //     const std::vector<senseAD::PredictionTrajectory>& trajectory_array =
    // //         frame_->model_obstacles_[i].trajectory_array;
    // //     const std::vector<senseAD::PredictionTrajectoryPoint>&
    // //         traj_point_array = trajectory_array[0].trajectory_point_array;
    // //     for (int j = 0; j + 1 < traj_point_array.size(); ++j) {
    // //         cv::Point2f grid_pt_0 = traj_point_array[j].position * 20.0;
    // //         cv::Point2f grid_pt_1 = traj_point_array[j + 1].position * 20.0;
    // //         cv::circle(grid_map_disp, grid_pt_0, 1, cv::Scalar(10, 215, 255),
    // //                    CV_FILLED);
    // //         cv::line(grid_map_disp, grid_pt_0, grid_pt_1,
    // //                  cv::Scalar(10, 215, 255), 0.1);
    // //     }
    // // }
    // // // 4.display boundary
    // // for (auto pop : frame_->boundary_obstacles_) {
    // //     std::vector<cv::Point2f> boundary = pop.LinePoints();
    // //     for (int i = 0; i + 1 < boundary.size(); ++i) {
    // //         cv::Point2f grid_pt_0 = boundary[i] * 20.0;
    // //         cv::Point2f grid_pt_1 = boundary[i + 1] * 20.0;
    // //         cv::circle(grid_map_disp, grid_pt_0, 0.2, cv::Scalar(0, 255, 255),
    // //                    CV_FILLED);
    // //         cv::line(grid_map_disp, grid_pt_0, grid_pt_1,
    // //                  cv::Scalar(0, 255, 255), 0.1);
    // //     }
    // // }
    // // 5.display refline
    // std::vector<cv::Point2f> ref_line;
    // for (int i = 0; i + 1 < target_trajectory_.cols(); ++i) {
    //     cv::Point2f ref_point;
    //     ref_point.x = target_trajectory_(0,i);
    //     ref_point.y = target_trajectory_(1,i);
    //     ref_line.push_back(ref_point);
    // }
    // for (int i = 0; i + 1 < ref_line.size(); ++i) {
    //     cv::Point2f grid_pt_0 = ref_line[i] * 20.0;
    //     cv::Point2f grid_pt_1 = ref_line[i + 1] * 20.0;
    //     // cv::circle(grid_map_disp, grid_pt_0, 0.2, cv::Scalar(0,
    //     // 255,
    //     // 255),
    //     //            CV_FILLED);
    //     cv::line(grid_map_disp, grid_pt_0, grid_pt_1, cv::Scalar(0, 69, 255), 1.0);
    // }
    // // 6. img flip
    // cv::Mat grid_map_disp_flip;
    // cv::flip(grid_map_disp, grid_map_disp_flip, 0);
    // // 7. display
    // // cv::imshow("trajectory_disp", grid_map_disp_flip);
    // cv::imwrite("../trajectory_disp.jpg", grid_map_disp_flip);
    // // cv::waitKey(-1);
    return;
}