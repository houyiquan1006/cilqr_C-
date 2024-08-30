/*
 * Copyright (C) 2022 by SenseTime Group Limited. All rights reserved.
 *
 */

#pragma once

#include <utility>
#include <vector>

namespace senseAD
{
    namespace perception
    {
        namespace camera
        {

            /*
             * @brief:
             * This class solve an optimization problem:
             * Y
             * |
             * |                       P(x1, y1)  P(x2, y2)
             * |            P(x0, y0)                       ... P(x(k-1), y(k-1))
             * |P(start)
             * |
             * |________________________________________________________ X
             *
             *
             * Given an initial set of points from 0 to k-1,  The goal is to find a set of
             * points which makes the line P(start), P0, P(1) ... P(k-1) "smooth".
             */
            struct FemPosDeviationSmootherConfig
            {
                double weight_fem_pos_deviation = 1.0e10;
                double weight_ref_deviation = 1.0;
                double weight_path_length = 1.0;
                bool apply_curvature_constraint = true;
                double weight_curvature_constraint_slack_var = 1.0e2;
                double curvature_constraint = 0.17;
                bool use_sqp = false;
                double sqp_ftol = 1e-4;
                double sqp_ctol = 1e-3;
                int sqp_pen_max_iter = 10;
                int sqp_sub_max_iter = 100;
                int max_iter = 500;
                double time_limit = 0.0;
                bool verbose = false;
                bool scaled_termination = true;
                bool warm_start = true;
                int print_level = 0;
                int max_num_of_iterations = 500;
                int acceptable_num_of_iterations = 15;
                double tol = 1e-8;
                double acceptable_tol = 1e-1;
            };
            class FemPosDeviationSmoother
            {
            public:
                explicit FemPosDeviationSmoother(
                    const FemPosDeviationSmootherConfig &config);

                bool Solve(const std::vector<std::pair<double, double>> &raw_point2d,
                           const std::vector<std::pair<double, double>> &bounds,
                           std::vector<double> *opt_x,
                           std::vector<double> *opt_y);

                bool QpWithOsqp(const std::vector<std::pair<double, double>> &raw_point2d,
                                const std::vector<std::pair<double, double>> &bounds,
                                std::vector<double> *opt_x,
                                std::vector<double> *opt_y);

                bool SqpWithOsqp(const std::vector<std::pair<double, double>> &raw_point2d,
                                 const std::vector<std::pair<double, double>> &bounds,
                                 std::vector<double> *opt_x,
                                 std::vector<double> *opt_y);

            private:
                FemPosDeviationSmootherConfig config_;
            };
        } // namespace camera
    } // namespace perception
} // namespace senseAD
