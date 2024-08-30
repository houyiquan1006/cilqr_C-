#pragma once

#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
enum class BoundaryDirection { 
    LEFT  =  1, 
    RIGHT = -1 
};

enum class Status {
    SUCCESS = 0,
    // the value parameter is invalid
    INVALID_PARAM = -1,
    // the input is nullptr
    NULL_PTR = -2,
    // general interal error
    INTERNAL_ERROR = -3,
    // map data is invalid
    MAP_DATA_ERROR = -4,
    // cannot find map element from hdmap
    MAP_FOUND_ERROR = -5,
    // the dimension of input is incompatible
    INCOMPATIBLE_DIMENSION = -6,
    // longitudinal search failed
    LONGI_SEARCH_FAILED = -7,
    // undefine error, when processing undefined enum
    UNDEFINED_ERROR = -8,
    // config reading error
    CONFIG_IO_ERROR = -9,
    // speed optimization fail
    SPEED_OPTIMIZER_FAIL = -10,
};

class BoundaryPoint {
public:
    // Cooridnate
    // Eigen::Vector2d pos;
    cv::Point2f pos;
    // Direction vector, [cos(theta), sin(theta)]
    // Eigen::Vector2d dir;
    cv::Point2f dir;
    // Type of boundary, left or right
    BoundaryDirection boundary_type;
};


class EgoPose {
public:
    double x = 0;
    double y = 0;
    double z = 0;
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
};

/**
 * @brief Trajectory point in prediction
 * 
 */
class PredictionTrajectoryPoint {
public:
    cv::Point2f position;
    cv::Point2f direction;
    cv::Point2f speed;
};

/**
 * @brief Prediction result with score and target road/lane info
 */
class PredictionTrajectory {
public:
    std::vector<PredictionTrajectoryPoint> trajectory_point_array;
    double score = 1.0; // the confidence of prediction
    // int target_road_id;
    // int target_lane_id;
    
};

/**
 * @brief Prediction result of a target object. 
 *        Every object has multiple prediction result, and is stored 
 *        as std:vector<PredictionTrajectory>.
 */
class PredictionObject {
public:
    // Object geometry info
    double length;
    double width;

    // Multiple prediction result
    std::vector<PredictionTrajectory> trajectory_array;
};
// struct TrajectoryPoint {
//     cv::Point2f position = cv::Point2f(0.0, 0.0);
//     cv::Point2f direction = cv::Point2f(0.0, 0.0);
//     double velocity = 0.0;
//     double acceleration = 0.0;
//     double jerk = 0.0;
//     double theta = 0.0;
//     double yaw_rate = 0.0;
//     double yaw_rate_dot = 0.0;
//     double curvature = 0.0;
//     double sum_distance = 0.0;
//     double time_difference = 0.0;
//     double kapparate = 0.0;
//     uint64_t timestamp = 0.0;  // unit: ns
// };
class TrajectoryPoint {
public:
    // Coordinate
    // Eigen::Vector2d pos;
    cv::Point2f position = cv::Point2f(0.0, 0.0);
    // Direction vector, [cos(theta), sin(theta)]
    cv::Point2f direction = cv::Point2f(0.0, 0.0);
    // Eigen::Vector2d dir;
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double theta = 0.0;
    double yaw_rate = 0.0;
    double yaw_rate_dot = 0.0;
    double curvature = 0.0;
    double sum_distance = 0.0;
    double time_difference = 0.0;
    double kapparate = 0.0;
    uint64_t timestamp = 0.0;  // unit: ns
};
struct Trajectory{
    uint64_t traj_base_timestamp_ns;
    std::vector<TrajectoryPoint> traj_point_array;
};
struct PlanningFrame{
    //std::vector<cv::Point2f> ref_line_;
    Trajectory ref_line_;
    TrajectoryPoint start_point_;
    std::vector<PredictionObject> model_obstacles_;
};