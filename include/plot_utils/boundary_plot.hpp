
#pragma once

#include "third_party/matplotlibcpp.hpp"
#include "common_utils/data_struct.hpp"

namespace plt = matplotlibcpp;

void plotBoundaries(const std::vector<std::vector<BoundaryPoint>>& boundary_set) {
    for (const auto& boundary : boundary_set) {
        std::vector<double> boundary_x;
        std::vector<double> boundary_y;
        std::string boundary_color = (boundary[0].boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

        for (const auto& boundary_point : boundary) {
            boundary_x.push_back(boundary_point.pos.x());
            boundary_y.push_back(boundary_point.pos.y());
        }

        plt::plot(boundary_x, 
                  boundary_y, 
                  {{"color", boundary_color}});
    }
}

void scatterEgo(const Eigen::MatrixXd& X_seq) {
    for (int i = 0; i < X_seq.cols(); i++) {
        Eigen::MatrixXd X = X_seq.col(i);
        plt::scatter(
            std::vector<double>{X(0, 0)},
            std::vector<double>{X(1, 0)},
            30,
            {{"color", "black"}}
        );
    }

    plt::axis("equal");
    plt::grid(true);
    plt::show();
}

void scatterBoundaryPoint(const BoundaryPoint& boundary_point) {
    std::string color = 
        (boundary_point.boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

    plt::scatter(
        std::vector<double>{boundary_point.pos.x()},
        std::vector<double>{boundary_point.pos.y()},
        30,
        {{"color", color}}
    );
}

void scatterBoundaryPoint(const Eigen::MatrixXd& pos,
                          BoundaryDirection type) {
    std::string color = 
        (type == BoundaryDirection::LEFT) ? "blue" : "red";

    plt::scatter(
        std::vector<double>{pos(0, 0)},
        std::vector<double>{pos(1, 0)},
        30,
        {{"color", color}}
    );    
}

void plotBoundaryPointQuiver(const BoundaryPoint& boundary_point) {
    std::string color = 
        (boundary_point.boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

    plt::quiver(
        std::vector<double>{boundary_point.pos.x()},
        std::vector<double>{boundary_point.pos.y()}, 
        std::vector<double>{boundary_point.dir.x()},
        std::vector<double>{boundary_point.dir.y()},
        {{"color", color}}               
    ); 
}

void plotBoundaryPointQuiver(const Eigen::MatrixXd& pos,
                             BoundaryPoint boundary_point) {
    std::string color = 
        (boundary_point.boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

    plt::quiver(
        std::vector<double>{pos(0, 0)},
        std::vector<double>{pos(1, 0)}, 
        std::vector<double>{boundary_point.dir.x()},
        std::vector<double>{boundary_point.dir.y()},
        {{"color", color}}               
    );         
}

void plotBoundaryPointQuiver(const Eigen::MatrixXd& pos,
                             const Eigen::MatrixXd& vector, 
                             BoundaryDirection type) {
    std::string color = 
        (type == BoundaryDirection::LEFT) ? "blue" : "red";

    plt::quiver(
        std::vector<double>{pos(0, 0)},
        std::vector<double>{pos(1, 0)}, 
        std::vector<double>{vector(0, 0)},
        std::vector<double>{vector(1, 0)},
        {{"color", color}}               
    );         
}

void plotEgoQuiver(const Eigen::MatrixXd& pos,
                   const Eigen::MatrixXd& vector) {
    plt::quiver(
        std::vector<double>{pos(0, 0)},
        std::vector<double>{pos(1, 0)}, 
        std::vector<double>{vector(0, 0)},
        std::vector<double>{vector(1, 0)},
        {{"color", "black"}}             
    );         
}

void plotEgoQuiver(const Eigen::MatrixXd& pos,
                   const Eigen::MatrixXd& vector,
                   const std::string& color) {
    plt::quiver(
        std::vector<double>{pos(0, 0)},
        std::vector<double>{pos(1, 0)}, 
        std::vector<double>{vector(0, 0)},
        std::vector<double>{vector(1, 0)},
        {{"color", color}}             
    );         
}