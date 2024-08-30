#include <gtest/gtest.h>

#include "common_utils/utils.hpp"
#include "common_utils/data_struct.hpp"
#include "cilqr_optimizer/cilqr_constructor.hpp"

int main() {
    // 读取数据
    std::string folder = "/home/qrq_20/sensetime/boundary_test/data";
    std::vector<std::vector<BoundaryPoint>> boundary_set;

    if (!readBoundarySetFromFolder(folder, boundary_set)) {
        std::cerr << "Read boundary failed" << std::endl;
        return 1;
    }

    // 绘制边界
    plotBoundaries(boundary_set);

    // 给定一个任意点，寻找 boundary 的最近邻以及投影到的最近线段
    Eigen::MatrixXd X_seq = Eigen::MatrixXd::Zero(4, 7);

    X_seq(0, 0) = -50;
    X_seq(1, 0) = -10;

    X_seq(0, 1) = -40;
    X_seq(1, 1) = -10;

    X_seq(0, 2) = -30;
    X_seq(1, 2) =  10;

    X_seq(0, 3) = -20;
    X_seq(1, 3) =  0;

    X_seq(0, 4) = -10;
    X_seq(1, 4) =  0;

    X_seq(0, 5) = 0;
    X_seq(1, 5) = 2;

    X_seq(0, 6) = 12;
    X_seq(1, 6) = 0;

    // 需要被验证的函数都在 CilqrConstructor，因此要先实例化
    // 一个 CilqrConstructor 对象
    CilqrConstructor cilqr_constructor;

    // 先寻找 X_seq 起点的最近点索引
    cilqr_constructor.findClosestRearBoundaryPointIndex(
        boundary_set, X_seq);

    // 测试 boundary segment 投影 & 原先的 closest boundary 函数
    cilqr_constructor.findClosestRearLineSegmentOnBoundarySet(
        boundary_set, X_seq);
    
    for (int i = 0; i < boundary_set.size(); i++) {
        std::vector<BoundaryPoint> boundary = boundary_set[i];
        
        // closest boundary points for X_seq
        Eigen::MatrixXd rear_closest_boundary_segments = 
            cilqr_constructor.rear_closest_boundary_segment_[i];

        // closest boundary vectors for X_seq
        Eigen::MatrixXd rear_closest_boundary_segment_vectors = 
            cilqr_constructor.rear_closest_boundary_segment_direction_[i];

        for (int j = 0; j < rear_closest_boundary_segments.cols(); j++) {
            // Head
            Eigen::MatrixXd segment_start_pos = rear_closest_boundary_segments.block(0, j, 2, 1);
            scatterBoundaryPoint(segment_start_pos, boundary[0].boundary_type);

            Eigen::MatrixXd segment_start_vector = rear_closest_boundary_segment_vectors.block(0, j, 2, 1);
            int type = static_cast<int>(rear_closest_boundary_segment_vectors(4, i));
            plotBoundaryPointQuiver(
                segment_start_pos,
                segment_start_vector, 
                static_cast<BoundaryDirection>(type));
            
            // Tail
            Eigen::MatrixXd segment_end_pos = rear_closest_boundary_segments.block(2, j, 2, 1);
            scatterBoundaryPoint(segment_end_pos, boundary[0].boundary_type);
            
            Eigen::MatrixXd segment_end_vector = rear_closest_boundary_segment_vectors.block(2, j, 2, 1);
            type = static_cast<int>(rear_closest_boundary_segment_vectors(4, i));
            plotBoundaryPointQuiver(
                segment_end_pos,
                segment_end_vector, 
                static_cast<BoundaryDirection>(type));

            // Normal vector
            Eigen::MatrixXd n_lambda;
            double ratio_lambda;
            Eigen::MatrixXd ego_pos = X_seq.block(0, j + 1, 2, 1);

            // Calculate signed distance
            double signed_dist = cilqr_constructor.getSignedDistanceToBoundary(
                ego_pos, 
                rear_closest_boundary_segments.block(0, j, 4, 1),
                rear_closest_boundary_segment_vectors.block(0, j, 5, 1),
                &n_lambda,
                &ratio_lambda
            );

            // calculate signed distance cost
            double signed_distance_cost = cilqr_constructor.getSignedDistanceBoundaryCost(
                0.5,
                0.5,    
                ego_pos, 
                rear_closest_boundary_segments.block(0, j, 4, 1),
                rear_closest_boundary_segment_vectors.block(0, j, 5, 1));

            std::cout << "signed distance = "      << signed_dist << std::endl;
            std::cout << "signed distance cost = " << signed_distance_cost << std::endl;

            // Calculate signed distance gradient
            Eigen::MatrixXd gradient = cilqr_constructor.getSignedDistanceBoundaryGrad(
                0.5,
                0.5,
                ego_pos,
                rear_closest_boundary_segments.block(0, j, 4, 1),
                rear_closest_boundary_segment_vectors.block(0, j, 5, 1));

            std::cout << "          " << " gradient: ("
                    << gradient(0, 0) << ", "
                    << gradient(1, 0) << ")\n"
                    << std::endl;

            plotBoundaries(boundary_set);
            // plotEgoQuiver(ego_pos, n_lambda);
            plotEgoQuiver(ego_pos, gradient);        
        }
    }

    // 从 Trajectory 的视角计算每个点的 Boundary Cost 和 Gradient
    double total_boundary_cost = 0;
    for (int i = 1; i < X_seq.cols(); i++) {
        // Boundary Cost
        double total_cost = cilqr_constructor.getStateCost(
            X_seq.col(i), X_seq.col(i), i);
        total_boundary_cost += total_cost;
        
        std::cout << "Step " << i << " state cost: " << total_cost << std::endl;

        // Boundary Cost gradient
        Eigen::MatrixXd total_gradient = cilqr_constructor.getStateCostGrad(
            X_seq.col(i), X_seq.col(i), i);

        std::cout << "      " << " gradient: ("
                  << total_gradient(0, 0) << ", "
                  << total_gradient(1, 0) << ")"
                  << std::endl;

        // Boundary cost hessian
        Eigen::MatrixXd total_hessian = cilqr_constructor.getStateCostHessian(
            X_seq.col(i), X_seq.col(i), i);

        plotEgoQuiver(X_seq.col(i), total_gradient, "green");        
        std::cout << "       " << "Hessian: " << std::endl;;

        for (int row = 0; row < total_hessian.rows(); row++) {
            std::cout << "       ";
            for (int col = 0; col < total_hessian.cols(); col++) {
                std::cout << total_hessian(row, col) 
                          << ", ";
            }    
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Total Boundary Cost =      " << total_boundary_cost << std::endl;

    // 计算累计的 Boundary Cost
    double total_boundary_cost_test = cilqr_constructor.getTotalBoundaryCost(X_seq);
    std::cout << "Total Boundary Cost Test = " << total_boundary_cost_test << std::endl;

    // 绘制自车位置的同时调用 plt::show()，绘制出所有的曲线
    plotBoundaries(boundary_set);
    scatterEgo(X_seq);

    return 0;
}

