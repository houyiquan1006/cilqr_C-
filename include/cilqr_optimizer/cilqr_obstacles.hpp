#pragma once

#include <vector>
#include "Eigen/Core"
#include "cilqr_optimizer/cilqr_config.hpp"
#include "common_utils/data_struct.hpp"
class CilqrObstacle {
 public:
    CilqrObstacle(double length, double width)
        : car_length_(length), car_width_(width) {}
    ~CilqrObstacle() {}

    /**
     * @brief Initialize ego vehicle info using config file
     * 
     * @param cfg Config file
     * @return Status
     */
    Status Init(const CilqrConfig& cfg);
    
    /**
     * @brief Get the gradient of obstacle constraints
     * 
     * @param q1 Param for barrier function
     * @param q2 Param for barrier function
     * @param X  Vehicle state sequence
     * @param i  Time step
     * @return Eigen::MatrixXd Gradient matrix
     */
    Eigen::MatrixXd getObstacleGrad(const double q1,
                                    const double q2,
                                    const Eigen::MatrixXd& X,
                                    const int i);
    
    /**
     * @brief Get the hessian matrix of obstacle constraints
     * 
     * @param q1 Param for barrier function
     * @param q2 Param for barrier function
     * @param X  Vehicle state sequence
     * @param i  Time step
     * @return Eigen::MatrixXd Hessian matrix 
     */
    Eigen::MatrixXd getObstacleHessian(const double q1,
                                       const double q2,
                                       const Eigen::MatrixXd& X,
                                       const int i);
    
    /**
     * @brief Get the barrier cost of obstacle constraints
     * 
     * @param q1 Param for barrier function
     * @param q2 Param for barrier function
     * @param X  Vehicle state sequence
     * @param i  Time step
     * @return double 
     */
    double getObstacleCost(const double q1,
                           const double q2,
                           const Eigen::MatrixXd& X,
                           const int i);
    
    /**
     * @brief Set the Transform Matrix object
     * 
     * @param i Time step
     */
    void setTransformMatrix();

    /**
     * @brief Set the predicted NPC trajectories
     * 
     * @param npc_states NPC states
     */
    void setNpcStates(const std::vector<std::vector<double>>& npc_states) {
        npc_states_.assign(npc_states.begin(), npc_states.end());
    }
    
    /**
     * @brief Get the Npc States Size object
     * 
     * @return int 
     */
    int getNpcStatesSize() { return static_cast<int>(npc_states_.size()); }

 private:
    int num_states_;    // Number of state variables
    int num_ctrls_;     // Number of control variables
    int horizon_;      // Prediction horizon
    double car_length_; // [m] Car length
    double car_width_;  // [m] Car width
    double t_safe_;     // [s] TTC safety buffer in obstacle moving direction
    double a_safe_;     // [m] Safety buffer in ellipse major axis
    double b_safe_;     // [m] Safety buffer in ellipse minor axis
    double r_safe_;     // [m] Safety buffer of vehicle collision shape

    std::vector<Eigen::MatrixXd> transform_P_;
    std::vector<std::vector<double>> npc_states_; 
};
