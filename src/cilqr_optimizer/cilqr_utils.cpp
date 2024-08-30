#include "cilqr_optimizer/cilqr_utils.hpp"

Eigen::MatrixXd getRotationMatrixFromGlobalToLocal(const Eigen::MatrixXd& point_p1,
                                                   const Eigen::MatrixXd& point_p2,
                                                   double* l) {
    // Check that the input points are 2D column vectors
    if (point_p1.rows() != 2 || point_p1.cols() != 1 ||
        point_p2.rows() != 2 || point_p2.cols() != 1) {
        throw std::invalid_argument("[getRotationMatrixFromGlobalToLocal] Input dimension mismatch!");
    }
    
    // Compute the shifted vector from p1 to p2
    Eigen::MatrixXd p2_shifted = point_p2 - point_p1;

    // Calculate the length of the vector
    *l = p2_shifted.norm();

    // Calculate the rotation matrix elements
    double cos_theta = p2_shifted(0, 0) / *l;
    double sin_theta = p2_shifted(1, 0) / *l;

    // Construct the rotation matrix
    Eigen::MatrixXd rotation_matrix(2, 2);
    rotation_matrix <<  cos_theta, sin_theta,
                       -sin_theta, cos_theta;

    return rotation_matrix;
}

Eigen::MatrixXd projectPointToSegment(const Eigen::MatrixXd& point_p, 
                                      const Eigen::MatrixXd& point_a,
                                      const Eigen::MatrixXd& point_b,
                                      double* ratio) {
    // Check that the input points are 2D column vectors
    if (point_p.rows() != 2 || point_p.cols() != 1 ||
        point_a.rows() != 2 || point_a.cols() != 1 ||
        point_b.rows() != 2 || point_b.cols() != 1) {
        std::cout << "column size: " << std::endl;
        std::cout << point_p.cols() << ", " 
                  << point_a.cols() << ", "
                  << point_b.cols() << ")" << std::endl;

        std::cout << "row size: " << std::endl;
        std::cout << point_p.rows() << ", " 
                  << point_a.rows() << ", "
                  << point_b.rows() << ")" << std::endl;

        throw std::invalid_argument("[projectPointToSegment] Input dimension mismatch!");

    }

    // Compute vectors ap and ab
    Eigen::MatrixXd ap = point_p - point_a;
    Eigen::MatrixXd ab = point_b - point_a;

    // Compute the denominator of the projection ratio
    double denominator = (ab.transpose() * ab)(0, 0);
    
    // Check for a valid denominator to avoid division by zero
    if (denominator == 0.0) {
        throw std::invalid_argument("[projectPointToSegment] The points A and B must not be the same.");
    }

    // Compute the projection ratio
    *ratio = (ap.transpose() * ab)(0, 0) / denominator;

    // Compute the projected point
    Eigen::MatrixXd projected_point = point_a + *ratio * ab;

    // Return the projected point
    return projected_point;    
}

double calculateEuclideanDistance(const Eigen::MatrixXd& point_a, 
                                  const Eigen::MatrixXd& point_b) {
    Eigen::VectorXd diff = point_b - point_a;
    return diff.norm();  
}

Eigen::Vector2d transformPoint(const Eigen::Vector2d& pointA,
                               const EgoPose& ego_pos_A,
                               const EgoPose& ego_pos_B) {
    double x_world = ego_pos_A.x + pointA.x() * cos(ego_pos_A.yaw) -
                     pointA.y() * sin(ego_pos_A.yaw);
    double y_world = ego_pos_A.y + pointA.x() * sin(ego_pos_A.yaw) +
                     pointA.y() * cos(ego_pos_A.yaw);

    double x_B = x_world - ego_pos_B.x;
    double y_B = y_world - ego_pos_B.y;

    double x_rotated = x_B * cos(ego_pos_B.yaw) + y_B * sin(ego_pos_B.yaw);
    double y_rotated = -x_B * sin(ego_pos_B.yaw) + y_B * cos(ego_pos_B.yaw);

    return {static_cast<double>(x_rotated), 
            static_cast<double>(y_rotated)};
}

Eigen::Vector3d transformPoint(const Eigen::Vector2d& pointA,
                               double thetaA,
                               const EgoPose& ego_pos_A,
                               const EgoPose& ego_pos_B) {
    // Transform thetaA to the ego_pos_B coordinate system
    double thetaA_in_B = thetaA + ego_pos_A.yaw - ego_pos_B.yaw;

    double x_world = ego_pos_A.x + pointA.x() * cos(ego_pos_A.yaw) -
                     pointA.y() * sin(ego_pos_A.yaw);
    double y_world = ego_pos_A.y + pointA.x() * sin(ego_pos_A.yaw) +
                     pointA.y() * cos(ego_pos_A.yaw);

    double x_B = x_world - ego_pos_B.x;
    double y_B = y_world - ego_pos_B.y;

    double x_rotated = x_B * cos(ego_pos_B.yaw) + y_B * sin(ego_pos_B.yaw);
    double y_rotated = -x_B * sin(ego_pos_B.yaw) + y_B * cos(ego_pos_B.yaw);

    return {static_cast<double>(x_rotated), 
            static_cast<double>(y_rotated),
            static_cast<double>(thetaA_in_B)};
}
void StateTransitfromReartoFront(Eigen::MatrixXd* X_front,
                                 const Eigen::MatrixXd& X_rear,
                                 int i,
                                 const double& wheelbase) {
    if (X_rear.rows() == 6) {
        // For vehicle model with X[x,y,v,theta,acc,yawrate] U[jerk,
        // yawrate_dot]
        (*X_front)(0, i) = X_rear(0, i) + cos(X_rear(3, i)) * wheelbase;
        (*X_front)(1, i) = X_rear(1, i) + sin(X_rear(3, i)) * wheelbase;
        (*X_front)(2, i) = X_rear(2, i);
        (*X_front)(3, i) = X_rear(3, i);
        (*X_front)(4, i) = X_rear(4, i);
        (*X_front)(5, i) = X_rear(5, i);
    } else if (X_rear.rows() == 5) {
        // For vehicle model with X[x,y,v,theta,acc] U[jerk, yawrate]
        (*X_front)(0, i) = X_rear(0, i) + cos(X_rear(3, i)) * wheelbase;
        (*X_front)(1, i) = X_rear(1, i) + sin(X_rear(3, i)) * wheelbase;
        (*X_front)(2, i) = X_rear(2, i);
        (*X_front)(3, i) = X_rear(3, i);
        (*X_front)(4, i) = X_rear(4, i);
    } else if (X_rear.rows() == 4) {
        // For vehicle model with X[x,y,theta,kappa] U[kappa_dot]
        (*X_front)(0, i) = X_rear(0, i) + cos(X_rear(2, i)) * wheelbase;
        (*X_front)(1, i) = X_rear(1, i) + sin(X_rear(2, i)) * wheelbase;
        (*X_front)(2, i) = X_rear(2, i);
        (*X_front)(3, i) = X_rear(3, i);
    } else {
        std::cerr << "wrong state dimension"<< std::endl;
    }
    return;
}