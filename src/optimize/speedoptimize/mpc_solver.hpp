/*
 * Copyright (C) 2023 by SenseTime Group Limited. All rights reserved.
 *
 */
#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include "mpc_osqp.hpp"

namespace senseAD {
namespace perception {
namespace speed {
class MpcSolver {
 public:
    using Ptr = std::shared_ptr<MpcSolver>;

    /**
     * @brief  construct mpc solver
     */
    MpcSolver() { osqp_solver_.reset(new MpcOsqp()); }

    /**
     * @brief solve mpc problem by using osqp solver
     *
     * @return
     */
    bool Solve() {
        MpcOsqp::Input input;
        this->GetConstraint(&input);
        this->GetCostFunction(&input);
        this->GetLinearizedModel(&input);
        this->GetConfig(&input);
        const auto& status = osqp_solver_->Solve(input, false, &solution_);
        return this->CheckSolution(status);
    }

 protected:
    /**
     * @brief get constraint
     *
     * @param input
     */
    virtual void GetConstraint(MpcOsqp::Input* input) const = 0;

    /**
     * @brief get cost function
     *
     * @param input
     */
    virtual void GetCostFunction(MpcOsqp::Input* input) const = 0;

    /**
     * @brief get linearized model
     *
     * @param input
     */
    virtual void GetLinearizedModel(MpcOsqp::Input* input) const = 0;

    /**
     * @brief get config
     *
     * @param input
     */
    virtual void GetConfig(MpcOsqp::Input* input) const = 0;

    /**
     * @brief check output of osqp solver
     *
     * @param status
     *
     * @return
     */
    inline bool CheckSolution(const MpcOsqp::Status& status) const {
        // status check
        if (status != MpcOsqp::Status::SUCCESS) {
            std::cerr << __LINE__ << solver_type_
                      << " mpc osqp solver status = "
                      << static_cast<int>(status);
            if (status < MpcOsqp::Status::SUCCESS) {
                std::cerr << __LINE__ << solver_type_ << " mpc solver failed";
                return false;
            }
        }

        // check nan
        for (const auto& sol : solution_) {
            if (std::isnan(sol)) {
                std::cerr << __LINE__ << solver_type_
                          << " mpc solver solution has nan";
                return false;
            }
        }

        return true;
    }

 protected:
    MpcOsqp::Ptr osqp_solver_ = nullptr;
    std::vector<double> solution_ = {};
    std::string solver_type_ = "";
};
}  // namespace camera
}  // namespace perception
}  // namespace senseAD
