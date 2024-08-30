// #pragma once

// #include <Eigen/Dense>
// #include <Eigen/Dense>

// #include "third_party/matplotlibcpp.hpp"

// namespace plt = matplotlibcpp;

// std::vector<double> eigenToVector(const Eigen::MatrixXd& matrix) {
//     std::vector<double> result;
//     result.reserve(matrix.rows());

//     for (int i = 0; i < matrix.rows(); i++) {
//         result.push_back(matrix(i, 0));
//     }

//     return result;
// }

// void plotCruiseData(const std::vector<double>& timeline,
//                     const std::vector<double>& s_forward_list,
//                     const std::vector<double>& v_forward_list,
//                     const std::vector<double>& a_forward_list,
//                     const std::vector<double>& jerk_forward_list) {
//     plt::figure_size(1000, 600);

//     // s-t
//     plt::subplot(2, 2, 1);
//     plt::plot(
//         timeline, 
//         s_forward_list, 
//         {{"label", "forward sim result"}}
//     );
//     plt::xlabel("time (s)");
//     plt::ylabel("accumulated length (m)");
//     plt::legend();
//     plt::grid(true);

//     // v-t
//     plt::subplot(2, 2, 2);
//     plt::plot(timeline, v_forward_list);
//     plt::xlabel("time (s)");
//     plt::ylabel("velocity (m/s)");
//     plt::grid(true);
//     plt::ylim(0, 33);

//     // a-t
//     plt::subplot(2, 2, 3);
//     plt::plot(timeline, a_forward_list);
//     plt::xlabel("time (s)");
//     plt::ylabel("accel (m/s^2)");
//     plt::grid(true);
//     plt::ylim(-2, 1);

//     // jerk-t
//     plt::subplot(2, 2, 4);
//     plt::plot(timeline, jerk_forward_list);
//     plt::xlabel("time (s)");
//     plt::ylabel("jerk (m/s^3)");
//     plt::grid(true);
//     plt::ylim(-4, 4);

//     // Display trigger
//     plt::show();
//     plt::save("/home/sensetime/ws/mini_cilqr_joint_planner/viz/cruise_result.png");
// };

// void plotCruiseData(const std::vector<double>& timeline,
//                     const Eigen::MatrixXd& s_forward_list,
//                     const Eigen::MatrixXd& v_forward_list,
//                     const Eigen::MatrixXd& a_forward_list,
//                     const Eigen::MatrixXd& jerk_forward_list) {
//     std::vector<double> s_forward = eigenToVector(s_forward_list);
//     std::vector<double> v_forward = eigenToVector(v_forward_list);
//     std::vector<double> a_forward = eigenToVector(a_forward_list);
//     std::vector<double> jerk_forward = eigenToVector(jerk_forward_list);

//     plotCruiseData(timeline, s_forward, v_forward, a_forward, jerk_forward);
// }

// void plotAccData(const std::vector<double>& timeline,
//                  const std::vector<double>& prediction_timeline,
//                  const std::vector<double>& s_obs_list,
//                  const std::vector<double>& s_forward_list,
//                  const std::vector<double>& v_forward_list,
//                  const std::vector<double>& a_forward_list,
//                  const std::vector<double>& jerk_forward_list) {
//     plt::figure_size(1000, 600);

//     // s-t
//     plt::subplot(2, 2, 1);
//     plt::plot(
//         timeline, 
//         s_forward_list, 
//         {{"label", "forward sim result"}}
//     );

//     plt::plot(
//         prediction_timeline, 
//         s_obs_list, 
//         {{"label", "predicted obs"}}
//     );

//     plt::xlabel("time (s)");
//     plt::ylabel("accumulated length (m)");
//     plt::legend();
//     plt::grid(true);

//     // v-t
//     plt::subplot(2, 2, 2);
//     plt::plot(timeline, v_forward_list);
//     plt::xlabel("time (s)");
//     plt::ylabel("velocity (m/s)");
//     plt::grid(true);
//     plt::ylim(0, 33);

//     // a-t
//     plt::subplot(2, 2, 3);
//     plt::plot(timeline, a_forward_list);
//     plt::xlabel("time (s)");
//     plt::ylabel("accel (m/s^2)");
//     plt::grid(true);
//     plt::ylim(-2, 1);

//     // jerk-t
//     plt::subplot(2, 2, 4);
//     plt::plot(timeline, jerk_forward_list);
//     plt::xlabel("time (s)");
//     plt::ylabel("jerk (m/s^3)");
//     plt::grid(true);
//     plt::ylim(-4, 4);

//     // Display trigger
//     plt::show();
//     plt::save("/home/sensetime/ws/mini_cilqr_joint_planner/viz/acc_result.png");
// };

// void plotAccData(const std::vector<double>& timeline,
//                  const std::vector<double>& prediction_timeline, 
//                  const std::vector<double>& s_obs_list,
//                  const Eigen::MatrixXd& s_forward_list,
//                  const Eigen::MatrixXd& v_forward_list,
//                  const Eigen::MatrixXd& a_forward_list,
//                  const Eigen::MatrixXd& jerk_forward_list) {
//     std::vector<double> s_forward = eigenToVector(s_forward_list);
//     std::vector<double> v_forward = eigenToVector(v_forward_list);
//     std::vector<double> a_forward = eigenToVector(a_forward_list);
//     std::vector<double> jerk_forward = eigenToVector(jerk_forward_list);

//     plotAccData(timeline, prediction_timeline, s_obs_list, s_forward, v_forward, a_forward, jerk_forward);
// }

// void plotRefAndTarget(const Eigen::MatrixXd& target,
//                       const std::vector<TrajectoryPoint>& ref,
//                       const Eigen::MatrixXd& X,
//                       const std::vector<double>& timeline) {
//     plt::figure_size(600, 600);
    
//     plt::subplot(2, 1, 1);
//     plt::scatter(
//         std::vector<double>{X(0, 0)}, 
//         std::vector<double>{X(1, 0)},
//         30,
//         {{"label", "Ego"}}
//     );

//     std::vector<double> ref_x;
//     std::vector<double> ref_y;
//     std::vector<double> target_x;
//     std::vector<double> target_y;
//     std::vector<double> target_v;

//     std::cout << "size of target: " << target.cols() << std::endl;

//     for (int i = 0; i < target.cols(); i++) {
//         target_x.push_back(target(0, i));
//         target_y.push_back(target(1, i));
//         target_v.push_back(target(2, i));

//         std::cout << " x: " << target(0, i) 
//                   << " y: " << target(1, i) << std::endl;    
//     }

//     plt::scatter(target_x, target_y, 20, {{"label", "target"}});

//     for (const auto& point : ref) {
//         ref_x.push_back(point.pos.x());
//         ref_y.push_back(point.pos.y());
//     }

//     plt::plot(ref_x, ref_y, {{"label", "ref line"}});
//     plt::legend();
//     plt::xlabel("X (m)");
//     plt::ylabel("Y (m)");

//     plt::subplot(2, 1, 2);
//     plt::plot(timeline, target_v);
//     plt::xlabel("time (s)");
//     plt::ylabel("velocity (m/s)");
    
//     plt::show();
// }