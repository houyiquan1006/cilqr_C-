#pragma once
#include <algorithm>
#include <cmath>
#include "common_utils/data_struct.hpp"
#include "cilqr_optimizer/cilqr_config.hpp"
#include "cilqr_optimizer/cilqr_constructor.hpp"
#include "cilqr_optimizer/cilqr_utils.hpp"
class CilqrJointPlanner
{
public:
    Status Init(CilqrConfig *cfg);
    Status Update(PlanningFrame *frame);
    Status setLocalPlan(std::vector<TrajectoryPoint> ref_line);
    Status setBoundary();
    Status findReferenceInfo();
    /**
     * @brief Calculate initial control sequence using result from last
     * iteration
     *
     * @param U_previous        [in ] control sequence in previous iteration

     * @return Status
     */
    Status setInitialSolution(const Eigen::MatrixXd &U_previous);
    /**
     * @brief Calculate target s-t trajectory in car following
     *        mode using forward simulation.
     *
     * @param ref_v          [in ] reference cruise speed
     * @param ego_init_v     [in ] initial velocity of ego vehicle
     * @param ego_init_a     [in ] initial acceleration of ego vehicle
     * @param v_max_list     [in ] maximum velocity constraints list
     * @param s_forward_list [out] target s sequence
     * @param v_forward_list [out] target v sequence
     */
    void findAccTargetSandV(double ref_v,
                            double ego_init_v,
                            double ego_init_a,
                            double v_max,
                            Eigen::MatrixXd *s_forward_list,
                            Eigen::MatrixXd *v_forward_list,
                            Eigen::MatrixXd *a_forward_list,
                            Eigen::MatrixXd *jerk_forward_list);

    /**
     * @brief Calculate target s-t trajectory in cruise mode using
     *        forward simulation.
     *
     * @param ref_v          [in ] reference cruise speed
     * @param ego_init_v     [in ] initial velocity of ego vehicle
     * @param ego_init_a     [in ] initial acceleration of ego vehicle
     * @param v_max_list     [in ] maximum velocity constraints list
     * @param s_forward_list [out] target s sequence
     * @param v_forward_list [out] target v sequence
     */
    void findCruiseTargetSandV(double ref_v,
                               double ego_init_v,
                               double ego_init_a,
                               double v_max,
                               Eigen::MatrixXd *s_forward_list,
                               Eigen::MatrixXd *v_forward_list,
                               Eigen::MatrixXd *a_forward_list,
                               Eigen::MatrixXd *jerk_forward_list);

    /**
     * @brief Calculate target s-t trajectory and find corresponding
     *        x-y-t trajectory
     *
     * @param ref_v      [in] reference velocity
     * @param v_max_list [in] maximum veclocity constraints list
     * @param X          [in] initial ego state
     */
    void findTargetTrajectory();

    Eigen::MatrixXd get_target_trajectory() { return target_trajectory_; }
    Status RunCilqrPlanner(Trajectory *out_path);
    Status setInitState();
    void vehicleModelLinearization();
    void vehicleModel(const int i,
                      Eigen::MatrixXd *X_ptr,
                      Eigen::MatrixXd *U_ptr);
    Status forwardPass();
    Status forwardPass(double alpha);
    Status backwardPass();
    /**
     * @brief Calculate optimal control sequence
     *
     * @param X_init Initial state sequence
     * @param U_init Initial control sequence
     * @return Status
     */
    Status getOptimalControlSeq();
    void setFallbackSolutionfromLastSolution();
    Status extractOutputTrajectory(Trajectory *out_path);
    Status checkTrajectoryStatus(Trajectory *out_path);
    bool isTrajectoryFeasible(const Trajectory& out_path);
    void normalizeHeading(double* heading);
    void visualizeTrajectory(const std::vector<TrajectoryPoint>& trajectory);
private:
    PlanningFrame *frame_;
    // Config class that stores and manages all related parameters
    CilqrConfig cfg_;
    // cilqr parameter
    std::vector<double> t_seq_;
    int iters_;
    double update_dt_;
    double dense_timestep_;
    double sparse_timestep_;
    int horizon_;
    int dense_horizon_;
    double tol_;
    int max_iter_;
    int min_iter_;
    int num_states_;
    int num_ctrls_;
    int num_auxiliary_;
    double interval_;
    double lambda_factor_;
    double max_lambda_;
    double lambda_;

    // Cost term
    Eigen::MatrixXd target_trajectory_;

    CilqrConstructor cilqr_constructor_;
    // Store current robot pose
    Eigen::MatrixXd x0_;
    // Eigen::MatrixXd u0_;
    // Store X U sequence
    Eigen::MatrixXd X_;
    Eigen::MatrixXd U_;
    Eigen::MatrixXd X_last_;
    Eigen::MatrixXd U_last_;
    Eigen::MatrixXd X_new_;
    Eigen::MatrixXd U_new_;

    std::vector<double> alphas_ = {1.0, 0.9, 0.68, 0.42, 0.21,
                                   0.092, 0.032, 0.1, 0.002, 0.0004};
    // cilqr data
    Eigen::MatrixXd l_x_;
    Eigen::MatrixXd l_u_;
    std::vector<Eigen::MatrixXd> l_xx_;
    std::vector<Eigen::MatrixXd> l_uu_;
    std::vector<Eigen::MatrixXd> l_ux_;
    std::vector<Eigen::MatrixXd> f_x_;
    std::vector<Eigen::MatrixXd> f_u_;
    Eigen::MatrixXd Q_x_;
    Eigen::MatrixXd Q_u_;
    Eigen::MatrixXd Q_xx_;
    Eigen::MatrixXd Q_uu_;
    Eigen::MatrixXd Q_ux_;
    // cilqr output
    Eigen::MatrixXd k_;
    std::vector<Eigen::MatrixXd> K_;

    // std::vector<double> vector_w_state_;
    // std::vector<double> vector_w_control_;
    // std::vector<double> vector_w_auxiliary_;
    // Environment info
    std::vector<TrajectoryPoint> local_plan_;

    // Prediction related
    const PredictionObject *predict_object_ptr_ = nullptr;
    double wheelbase_;
};