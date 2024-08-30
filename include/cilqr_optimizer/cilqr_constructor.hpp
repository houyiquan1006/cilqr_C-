#pragma once

#include <vector>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "common_utils/data_struct.hpp"
#include "cilqr_optimizer/cilqr_config.hpp"
#include "cilqr_optimizer/cilqr_obstacles.hpp"
class CilqrConstructor {
 public:
    // cppcheck-suppress uninitMemberVar
    CilqrConstructor() {}
    ~CilqrConstructor() {}

    // init constraints
    Status Init(CilqrConfig* cfg,
                const std::vector<PredictionObject>& model_obstacles);
    std::vector<CilqrObstacle> getObs() {
        return obs_constraints_;
    }
    // set obstacles
    void setObsConstraints(
        const CilqrConfig& cfg,
        const std::vector<PredictionObject>& model_obstacles);

    // calculate cost grad and hessian
    // void findTrajectoryStartIndex(
    //     const std::vector<senseAD::planning::ENUPose>& local_plan,
    //     const Eigen::MatrixXd& X);
    // void findTargetTrajectory(
    //     const std::vector<senseAD::planning::ENUPose>& local_plan,
    //     const Eigen::MatrixXd& X_seq);
    void setTargetTrajectory(const Eigen::MatrixXd& target_trajectory) {
        target_trajectory_ = target_trajectory;
    }
    // void findLastTrajectoryStartIndex(
    //     const std::vector<senseAD::planning::ENUPose>& last_plan,
    //     const Eigen::MatrixXd& X);
    // void findLastTargetTrajectory(
    //     const std::vector<senseAD::planning::ENUPose>& last_plan,
    //     const Eigen::MatrixXd& X_seq);
    void setLastTargetTrajectory(
        const Eigen::MatrixXd& target_last_trajectory) {
        target_last_trajectory_ = target_last_trajectory;
    }

    void setRefinedConstraints(
        const std::vector<Eigen::MatrixXd>& mul_frame_state_constraints_min,
        const std::vector<Eigen::MatrixXd>& mul_frame_state_constraints_max) {
        mul_frame_state_constraints_min_ = mul_frame_state_constraints_min;
        mul_frame_state_constraints_max_ = mul_frame_state_constraints_max;
    }

    void findRearBoundaryStartIndex(
        const std::vector<std::vector<BoundaryPoint>>& boundary,
        const Eigen::MatrixXd& X);
    void findFrontBoundaryStartIndex(
        const std::vector<std::vector<BoundaryPoint>>& boundary,
        const Eigen::MatrixXd& X);
    void findRearClosestBoundary(
        const std::vector<std::vector<BoundaryPoint>>& boundary,
        const Eigen::MatrixXd& X_seq);
    void findFrontClosestBoundary(
        const std::vector<std::vector<BoundaryPoint>>& boundary,
        const Eigen::MatrixXd& X_seq);

    Eigen::MatrixXd getStateCostGrad(const Eigen::MatrixXd& X,
                                     const Eigen::MatrixXd& X_last,
                                     const int i);
    Eigen::MatrixXd getStateCostHessian(const Eigen::MatrixXd& X, const int i);
    Eigen::MatrixXd getControlCostGrad(const Eigen::MatrixXd& U,
                                       const Eigen::MatrixXd& U_last);
    Eigen::MatrixXd getControlCostHessian(const Eigen::MatrixXd& U);
    double barrierFunction(const double q1, const double q2, const double c);
    double barrierFunction(const Eigen::MatrixXd& q1,
                           const Eigen::MatrixXd& q2,
                           const Eigen::MatrixXd& c);
    Eigen::MatrixXd barrierFunctionGrad(const double q1,
                                        const double q2,
                                        const double c,
                                        const Eigen::MatrixXd& c_grad);
    Eigen::MatrixXd barrierFunctionGrad(const Eigen::MatrixXd& q1,
                                        const Eigen::MatrixXd& q2,
                                        const Eigen::MatrixXd& c,
                                        const Eigen::MatrixXd& c_grad);
    Eigen::MatrixXd barrierFunctionHessian(const double q1,
                                           const double q2,
                                           const double c,
                                           const Eigen::MatrixXd& c_grad);
    Eigen::MatrixXd barrierFunctionHessian(const Eigen::MatrixXd& q1,
                                           const Eigen::MatrixXd& q2,
                                           const Eigen::MatrixXd& c,
                                           const Eigen::MatrixXd& c_grad);
    double getBoundaryCost(
        const double q1,
        const double q2,
        const Eigen::MatrixXd& X,
        const int i,
        const int j,
        const std::vector<Eigen::MatrixXd>& closest_boundary,
        const std::vector<Eigen::MatrixXd>& closest_boundary_direction);
    Eigen::MatrixXd getBoundaryGrad(
        const double q1,
        const double q2,
        const Eigen::MatrixXd& X,
        const int i,
        const int j,
        const std::vector<Eigen::MatrixXd>& closest_boundary,
        const std::vector<Eigen::MatrixXd>& closest_boundary_direction);
    Eigen::MatrixXd getBoundaryHessian(
        const double q1,
        const double q2,
        const Eigen::MatrixXd& X,
        const int i,
        const int j,
        const std::vector<Eigen::MatrixXd>& closest_boundary,
        const std::vector<Eigen::MatrixXd>& closest_boundary_direction);
    Eigen::MatrixXd getAuxiliaryCostWeight() { return auxiliary_cost_;}
    double getAuxiliaryCost(const Eigen::MatrixXd& X,
                            const Eigen::MatrixXd& U,
                            const int i);
    Eigen::MatrixXd getAuxiliaryCostGrad(const Eigen::MatrixXd& X,
                                const Eigen::MatrixXd& U,
                                const int i);
    Eigen::MatrixXd getAuxiliaryCostHessian(const Eigen::MatrixXd& X,
                                   const Eigen::MatrixXd& U,
                                   const int i);
    double getStateCost(const Eigen::MatrixXd& X,
                        const Eigen::MatrixXd& X_last,
                        const int i);
    double getControlCost(const Eigen::MatrixXd& U,
                          const Eigen::MatrixXd& U_last);
    std::tuple<double,double,double> getTotalTraCost(const Eigen::MatrixXd& X_seq);
    double getTotalStateRegCost(const Eigen::MatrixXd& X_seq);
    double getTotalBoundaryCost(const Eigen::MatrixXd& X_seq);
    double getTotalStateBarrierCost(const Eigen::MatrixXd& X_seq);
    void getTotalCost(const Eigen::MatrixXd& X_seq,
                      const Eigen::MatrixXd& U_seq,
                      const Eigen::MatrixXd& X_seq_last,
                      const Eigen::MatrixXd& U_seq_last,
                      double* total_cost,
                      double* total_state_cost,
                      double* total_control_cost,
                      double* total_auxiliary_cost);
    Eigen::MatrixXd getTargetref() { return target_trajectory_; }

    std::vector<Eigen::MatrixXd> getFrontClosestBoundaries() {
        return front_closest_boundary_;
    }
    std::vector<Eigen::MatrixXd> getRearClosestBoundaries() {
        return rear_closest_boundary_;
    }

    std::vector<Eigen::MatrixXd> getRearClosestBoundarySegments() {
        return rear_closest_boundary_segment_;
    }

    std::vector<Eigen::MatrixXd> getFrontClosestBoundarySegments() {
        return front_closest_boundary_segment_;
    }

    std::vector<Eigen::MatrixXd> getRearClosestBoundarySegmentDirections() {
        return rear_closest_boundary_segment_direction_;
    }

    std::vector<Eigen::MatrixXd> getFrontClosestBoundarySegmentDirections() {
        return front_closest_boundary_segment_direction_;
    }

    /**
     * @brief 在一条 Boundary 中当前车辆位置的最近 line segment，记录
     *        line segment 的端点坐标以及端点切向量
     * 
     * @param boundary                  [in ] 离散的边界点
     * @param X                         [in ] 自车位置
     * @param closest_idx               [in ] 距离自车最近的边界点索引
     * @param closest_segment_start     [out] line segment 起点坐标
     * @param closest_segment_end       [out] line segment 终点坐标
     * @param closest_segment_start_dir [out] line segment 起点切向量 
     * @param closest_segment_end_dir   [out] line segment 重点切向量
     */
    void findClosestLineSegmentOnSingleBoundary(
        const std::vector<BoundaryPoint>& boundary,
        const Eigen::MatrixXd& X,
        const int closest_idx,
        Eigen::MatrixXd* closest_segment_start,
        Eigen::MatrixXd* closest_segment_end,
        Eigen::MatrixXd* closest_segment_start_dir,
        Eigen::MatrixXd* closest_segment_end_dir);

    /**
     * @brief 给定 Trajectory 序列，每个状态量在 Boundary Set 上的
     *        最近 Line Segment
     * 
     * @param boundary  vector of boundary points
     * @param X_seq     sequence of state
     * @param axle_type rear or front
     */

    void findClosestLineSegmentOnBoundarySet(
        const std::vector<std::vector<BoundaryPoint>>& boundary,
        const Eigen::MatrixXd& X_seq,
        const std::string& axle_type);

    /**
     * @brief Get the signed distance to given boundary segment
     * 
     * @param X                                  [in ] Ego position
     * @param closest_boundary_segment           [in ] Closest line segment head 
     *                                                 and tail position
     * @param closest_boundary_segment_direction [in ] Tangential vector of line
     *                                                 segment head and tail.
     * @param n_lambda                           [out] Normal vector pointing at
     *                                                 line segment.
     * @param ratio_lambda                       [out] Projection ratio.
     * @return double 
     */
    double getSignedDistanceToBoundary(
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& closest_boundary_segment,
        const Eigen::MatrixXd& closest_boundary_segment_direction,
        Eigen::MatrixXd* n_lambda,
        double* ratio_lambda);

    /**
     * @brief Calculate boundary cost with barrier penalty g[c(x)]
     * 
     * @param q1                                 [in] barrier function param
     * @param q2                                 [in] barrier function param
     * @param X                                  [in] Ego position
     * @param closest_boundary_segment           [in] Closest line segment head 
     *                                                and tail position
     * @param closest_boundary_segment_direction [in] Tangential vector of line
     *                                                segment head and tail.
     * @return double 
     */
    double getSignedDistanceBoundaryCost(
        const double q1,
        const double q2,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& closest_boundary_segment,
        const Eigen::MatrixXd& closest_boundary_segment_direction);

    /**
     * @brief Gradient of signed distance boundary constraint.
     */
    Eigen::MatrixXd getSignedDistanceBoundaryGrad(
        const double q1,
        const double q2,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& closest_boundary_segment,
        const Eigen::MatrixXd& closest_boundary_segment_direction,
        const std::string& axle_type);
    /**
     * @brief Hessian of signed distance boundary constraint. 
     */
    Eigen::MatrixXd getSignedDistanceBoundaryHessian(
        const double q1,
        const double q2,
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& closest_boundary_segment,
        const Eigen::MatrixXd& closest_boundary_segment_direction,
        const std::string& axle_type);
    const double PosWeightFromCurvatureMap[3][2] = {
        {0.001, 1.0},  // radius = 1000m
        {0.005, 1.0},  // radius = 200m
        {0.014, 2.0},  // radius = 70m
    };
    const double LatTermWeightFromCurvatureMap[3][2] = {
        {0.001, 1.0},  // radius = 1000m
        {0.005, 1.0},  // radius = 200m
        {0.014, 0.1},  // radius = 70m
    };

 private:
    /****** Switch between different config ******/

    int horizon_;
    int dense_horizon_;
    int num_states_;
    int num_ctrls_;
    int num_auxiliary_;
    double wheelbase_;
    double boundary_safe_;
    // double kappa_max_limit_;
    // double kappa_min_limit_;
    // double kapparate_max_limit_;
    // double kapparate_min_limit_;
    // double q1_boundary_;
    // double q2_boundary_;
    // double q1_kappa_;
    // double q2_kappa_;
    // double q1_kapparate_;
    // double q2_kapparate_;
    // double q1_front_;
    // double q2_front_;
    // double q1_rear_;
    // double q2_rear_;

    // *******Placeholder for different state and control space************* //
    Eigen::MatrixXd state_constraints_min_;
    Eigen::MatrixXd state_constraints_max_;
    Eigen::MatrixXd control_constraints_min_;
    Eigen::MatrixXd control_constraints_max_;

    std::vector<Eigen::MatrixXd> mul_frame_state_constraints_min_;
    std::vector<Eigen::MatrixXd> mul_frame_state_constraints_max_;

    Eigen::MatrixXd q1_states_;
    Eigen::MatrixXd q2_states_;
    Eigen::MatrixXd q1_controls_;
    Eigen::MatrixXd q2_controls_;

    //******************************************************************************//
    double q1_front_;
    double q2_front_;
    double q1_rear_;
    double q2_rear_;
    double q1_boundary_;
    double q2_boundary_;
    Eigen::MatrixXd control_cost_;
    Eigen::MatrixXd control_regularization_cost_;
    Eigen::MatrixXd state_cost_;  // State error weight matrix
    Eigen::MatrixXd state_regularization_cost_;
    Eigen::MatrixXd auxiliary_cost_;

    Eigen::MatrixXd P_barrier_states_;
    Eigen::MatrixXd P_barrier_controls_;
    Eigen::MatrixXd P_states_;
    Eigen::MatrixXd P_controls_;

    // int target_trajectory_start_index_;
    Eigen::MatrixXd target_trajectory_;
    // int target_last_trajectory_start_index_;
    Eigen::MatrixXd target_last_trajectory_;
    std::vector<int> rear_closest_boundary_start_index_;
    std::vector<int> front_closest_boundary_start_index_;
    std::vector<int> rear_closest_boundary_distances_;
    std::vector<Eigen::MatrixXd> rear_closest_boundary_;
    std::vector<Eigen::MatrixXd> rear_closest_boundary_direction_;
    std::vector<Eigen::MatrixXd> front_closest_boundary_;
    std::vector<Eigen::MatrixXd> front_closest_boundary_direction_;
    std::vector<CilqrObstacle> obs_constraints_;

    // 记录车辆当前后轴中心位置的最近 boundary line segment
    std::vector<Eigen::MatrixXd> rear_closest_boundary_segment_;

    // 记录车辆当前后轴中心位置的最近 boundary line segment 端点切向量
    std::vector<Eigen::MatrixXd> rear_closest_boundary_segment_direction_;

    // 记录车辆当前前轴中心位置的最近 boundary line segment
    std::vector<Eigen::MatrixXd> front_closest_boundary_segment_;

    // 记录车辆当前前轴中心位置的最近 boundary line segment 端点切向量
    std::vector<Eigen::MatrixXd> front_closest_boundary_segment_direction_;
};
