/*
 * Copyright (C) 2023 by SenseTime Group Limited. All rights reserved.
 *
 */
#pragma once

#include <vector>
#include <memory>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"
#include "mpc_solver.hpp"
#include "mpc_osqp.hpp"

namespace senseAD {
namespace perception {
using namespace senseAD::perception::speed;

namespace camera {
class SpeedMpcSolver : public MpcSolver {
 public:
    using Ptr = std::shared_ptr<SpeedMpcSolver>;

    /**
     * @brief construct path mpc solver
     */
    explicit SpeedMpcSolver(const OptimizeHorizon& horizon);
    SpeedMpcSolver(){};
    /**
     * @brief update
     *
     * @param initial_state
     * @param profile
     */
    void Update(const SpeedOptimizeState& initial_state,
                const std::vector<SpeedOptimizeProfile>& profile,
                const SpeedOptimizeWeight& weight);

    /**
     * @brief get solution
     *
     * @param states
     * @param controls
     */
    void GetSolution(std::vector<SpeedOptimizeState>* states,
                     std::vector<SpeedOptimizeControl>* controls) const;

 protected:
    /**
     * @brief get constraint
     *
     * @param input
     */
    void GetConstraint(MpcOsqp::Input* input) const override;

    /**
     * @brief get cost function
     *
     * @param input
     */
    void GetCostFunction(MpcOsqp::Input* input) const override;

    /**
     * @brief get linearized model
     *
     * @param input
     */
    void GetLinearizedModel(MpcOsqp::Input* input) const override;

    /**
     * @brief get config
     *
     * @param input
     */
    void GetConfig(MpcOsqp::Input* input) const override;

 private:
    /**
     * @brief get linearized model coeff
     *
     * @param discretize_order
     * @param interval_seq
     * @param matrix_coeff_vec
     *
     * @return
     */
    bool GetLinearizedModelCoeff(
        const int discretize_order,
        const std::vector<double>& interval_seq,
        std::vector<Eigen::MatrixXd>* matrix_coeff_vec) const;

 private:
    SpeedOptimizeState initial_state_{};
    std::vector<SpeedOptimizeProfile> profile_ = {};
    SpeedOptimizeWeight weight_{};

    // horizon
    const int state_dim_ = 3;
    const int control_dim_ = 1;
    const int state_constraint_dim_ = 3;
    const int state_cost_function_dim_ = 9;
    const int control_constraint_dim_ = 2;
    const int horizon_step_ = 10;
};
}  // namespace camera
}  // namespace perception
}  // namespace senseAD
