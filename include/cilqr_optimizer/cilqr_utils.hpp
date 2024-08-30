#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <stdexcept>

#include "common_utils/data_struct.hpp"

/**
 * @brief Calculate rotation matrix from local frame p1->p2 
 *        to local frame (0, 0) -> (l, 0). Designed for signed
 *        distance calculation. 
 * 
 * @param point_p1 [in ] Line segment start 
 * @param point_p2 [in ] Line segment end
 * @param l        [out] Distance from p1 to p2
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd getRotationMatrixFromGlobalToLocal(
    const Eigen::MatrixXd& point_p1,
    const Eigen::MatrixXd& point_p2,
    double* l);

/**
 * @brief Project point p to line segment ab.
 * 
 * @param point_p [in ] Point to be projected
 * @param point_a [in ] Line segment start
 * @param point_b [in ] Line segment end
 * @param ratio   [out] Projection ratio
 * @return Eigen::MatrixXd Projected point
 */
Eigen::MatrixXd projectPointToSegment(
    const Eigen::MatrixXd& point_p, 
    const Eigen::MatrixXd& point_a,
    const Eigen::MatrixXd& point_b,
    double* ratio);

/**
 * @brief Calculate euclidean distance between two point
 *
 * @param point_a [in] Location of point a
 * @param point_b [in] Location of point b
 * @return double Euclidean distance
 */
double calculateEuclideanDistance(
    const Eigen::MatrixXd& point_a, 
    const Eigen::MatrixXd& point_b);

/**
 * @brief transformPoint transform the coordinate of a point in A frame to B
 * frame given the coordinate in A frame, the transformation from frame A to the
 * global frame and the transformation from frame B to the global frame
 *
 * output is the coordinate of point in frame B
 */
Eigen::Vector2d transformPoint(const Eigen::Vector2d &pointA,
                               const EgoPose &ego_pos_A,
                               const EgoPose &ego_pos_B);

Eigen::Vector3d transformPoint(const Eigen::Vector2d& pointA,
                               double thetaA,
                               const EgoPose& ego_pos_A,
                               const EgoPose& ego_pos_B);

void StateTransitfromReartoFront(Eigen::MatrixXd* X_front,
                                 const Eigen::MatrixXd& X_rear,
                                 int i,
                                 const double& wheelbase);