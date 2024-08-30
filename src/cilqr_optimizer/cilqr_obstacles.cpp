#include "cilqr_optimizer/cilqr_obstacles.hpp"

Status CilqrObstacle::Init(const CilqrConfig& cfg) {
    num_states_ = cfg.ilqr.num_states;
    num_ctrls_ = cfg.ilqr.num_ctrls;
    horizon_ = cfg.ilqr.horizon;
    t_safe_ = cfg.constraint.obstacle.t_safe;
    a_safe_ = cfg.constraint.obstacle.a_safe;
    b_safe_ = cfg.constraint.obstacle.b_safe;
    r_safe_ = cfg.constraint.obstacle.r_safe;
    return Status::SUCCESS;
}

void CilqrObstacle::setTransformMatrix() {
    transform_P_.resize(horizon_ + 1);
    for (int i = 0; i < transform_P_.size(); ++i) {
        if (i >= npc_states_.size()) {
            transform_P_[i] = Eigen::MatrixXd::Zero(num_states_, num_states_);
            continue;
        }
        double v = npc_states_[i][2];
        double theta = npc_states_[i][3];
        double a = car_length_ + 2 * (a_safe_ + v * t_safe_ + r_safe_);
        double b = car_width_ + 2 * (b_safe_ + r_safe_);
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(num_states_, num_states_);
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(num_states_, num_states_);
        P(0, 0) = 1 / pow(a, 2);
        P(1, 1) = 1 / pow(b, 2);
        T(0, 0) = cos(theta);
        T(0, 1) = sin(theta);
        T(1, 0) = -sin(theta);
        T(1, 1) = cos(theta);
        transform_P_[i] = T.transpose() * P * T;
    }
    return;
}
Eigen::MatrixXd CilqrObstacle::getObstacleGrad(const double q1,
                                               const double q2,
                                               const Eigen::MatrixXd& X,
                                               const int i) {
    Eigen::MatrixXd obstacle_grad;
    Eigen::MatrixXd Xo;
    Xo = Eigen::MatrixXd::Zero(num_states_, 1);

    Xo(0, 0) = npc_states_[i][0];
    Xo(1, 0) = npc_states_[i][1];
    Xo(2, 0) = npc_states_[i][2];
    Xo(3, 0) = npc_states_[i][3];
    Xo(4, 0) = npc_states_[i][4];
    Xo(5, 0) = npc_states_[i][5];
    Eigen::MatrixXd tmp;
    tmp = (X - Xo).transpose() * transform_P_[i] * (X - Xo);
    double c = 1 - tmp(0, 0);
    obstacle_grad = -2 * q1 * q2 * exp(q2 * c) * transform_P_[i] * (X - Xo);
    return obstacle_grad;
}

Eigen::MatrixXd CilqrObstacle::getObstacleHessian(const double q1,
                                                  const double q2,
                                                  const Eigen::MatrixXd& X,
                                                  const int i) {
    Eigen::MatrixXd obstacle_hessian;
    Eigen::MatrixXd Xo;
    Xo = Eigen::MatrixXd::Zero(num_states_, 1);
    Xo(0, 0) = npc_states_[i][0];
    Xo(1, 0) = npc_states_[i][1];
    Xo(2, 0) = npc_states_[i][2];
    Xo(3, 0) = npc_states_[i][3];
    Xo(4, 0) = npc_states_[i][4];
    Xo(5, 0) = npc_states_[i][5];
    Eigen::MatrixXd tmp;
    tmp = (X - Xo).transpose() * transform_P_[i] * (X - Xo);
    double c = 1 - tmp(0, 0);
    obstacle_hessian = 4 * q1 * q2 * q2 * exp(q2 * c) * transform_P_[i] *
                           (X - Xo) * (X - Xo).transpose() * transform_P_[i] -
                       2 * q1 * q2 * exp(q2 * c) * transform_P_[i];
    return obstacle_hessian;
}

double CilqrObstacle::getObstacleCost(const double q1,
                                      const double q2,
                                      const Eigen::MatrixXd& X,
                                      const int i) {
    double obstacle_cost;
    Eigen::MatrixXd Xo;
    Xo = Eigen::MatrixXd::Zero(num_states_, 1);
    Xo(0, 0) = npc_states_[i][0];
    Xo(1, 0) = npc_states_[i][1];
    Xo(2, 0) = npc_states_[i][2];
    Xo(3, 0) = npc_states_[i][3];
    Xo(4, 0) = npc_states_[i][4];
    Xo(5, 0) = npc_states_[i][5];
    Eigen::MatrixXd tmp;
    tmp = (X - Xo).transpose() * transform_P_[i] * (X - Xo);
    double c = 1 - tmp(0, 0);
    // std::cout << "X: " << X << std::endl;
    // std::cout << "Xo: " << Xo << std::endl;
    // std::cout << "c: " << c << std::endl;
    obstacle_cost = q1 * exp(q2 * c);
    return obstacle_cost;
}