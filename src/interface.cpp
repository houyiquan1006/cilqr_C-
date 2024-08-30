#include <dirent.h>
#include <cstring>
#include "common_utils/utils.hpp"
#include "cilqr_joint_planner/cilqr_joint_planner.hpp"
#include "third_party/matplotlibcpp.hpp"
#include "plot_utils/acc_and_cruise_plot.hpp"
#include "interface.h"
#include "matplotlibcpp.hpp"
namespace plt = matplotlibcpp;
bool RoughSmooth(Traj &traj)
{
    senseAD::perception::camera::FemPosDeviationSmootherConfig fem_pos_config;
    fem_pos_config.apply_curvature_constraint = false;
    // fem_pos_config.weight_curvature_constraint_slack_var = 1000.0;
    senseAD::perception::camera::FemPosDeviationSmoother smoother(fem_pos_config);
    std::vector<std::pair<double, double>> raw_point2d, bounds;
    int pos_num = traj.size();
    raw_point2d.resize(pos_num);
    bounds.resize(pos_num);
    for (int i = 0; i < pos_num; ++i)
    {
        raw_point2d[i].first = traj[i].x;
        raw_point2d[i].second = traj[i].y;
        bounds[i].first = 0.35;
        bounds[i].second = 0.15;
    }
    bounds.front().first = 0.05;
    bounds.front().second = 0.05;
    // box contraints on pos are used in fem pos smoother, thus shrink
    // the bounds by 1.0 / sqrt(2.0)
    std::vector<std::pair<double, double>> box_bounds = bounds;
    const double box_ratio = 1.0 / std::sqrt(2.0);
    for (auto &bound : box_bounds)
    {
        bound.first *= box_ratio;
        bound.second *= box_ratio;
    }

    std::vector<double> opt_x;
    std::vector<double> opt_y;
    bool status = smoother.Solve(raw_point2d, box_bounds, &opt_x, &opt_y);
    if (status)
    {
        for (int i = 0; i < pos_num; ++i)
        {
            traj[i].x = opt_x[i];
            traj[i].y = opt_y[i];
        }
    }
    return status;
}
bool testsmooth(Traj &traj)
{
    senseAD::perception::camera::FemPosDeviationSmootherConfig fem_pos_config;
    fem_pos_config.apply_curvature_constraint = true;
    // fem_pos_config.weight_curvature_constraint_slack_var = 1000.0;
    senseAD::perception::camera::FemPosDeviationSmoother smoother(fem_pos_config);
    std::vector<std::pair<double, double>> raw_point2d, bounds;
    int pos_num = traj.size();
    raw_point2d.resize(pos_num);
    bounds.resize(pos_num);
    for (int i = 0; i < pos_num; ++i)
    {
        raw_point2d[i].first = traj[i].x;
        raw_point2d[i].second = traj[i].y;
        bounds[i].first = 0.05;
        bounds[i].second = 0.05;
    }
    bounds.front().first = 0.05;
    bounds.front().second = 0.05;
    // box contraints on pos are used in fem pos smoother, thus shrink
    // the bounds by 1.0 / sqrt(2.0)
    std::vector<std::pair<double, double>> box_bounds = bounds;
    const double box_ratio = 1.0 / std::sqrt(2.0);
    for (auto &bound : box_bounds)
    {
        bound.first *= box_ratio;
        bound.second *= box_ratio;
    }

    std::vector<double> opt_x;
    std::vector<double> opt_y;
    bool status = smoother.Solve(raw_point2d, box_bounds, &opt_x, &opt_y);
    return status;
}
bool testcilqr(Traj &traj)
{
    PlanningFrame frame;
    Trajectory ref_line;
    ref_line.traj_point_array.resize(traj.size());
    for (int i = 0; i < traj.size(); ++i)
    {
        ref_line.traj_point_array[i].position.x = traj[i].x;
        ref_line.traj_point_array[i].position.y = traj[i].y;
    }
    frame.ref_line_ = ref_line;
    // ideal update
    float ego_v = 5.0f;
    float ego_a = 0.0f;
    float ego_yaw_rate = 0.0f;
    std::vector<PredictionObject> obstacles;
    frame.start_point_.position = cv::Point2f(0., 0.);
    frame.start_point_.direction = cv::Point2f(1., 0.);
    frame.start_point_.velocity = ego_v;
    frame.start_point_.acceleration = ego_a;
    frame.start_point_.yaw_rate = ego_yaw_rate;

    frame.model_obstacles_ = obstacles;
    // Create planner
    CilqrJointPlanner cilqr_joint_planner;
    CilqrConfig cfg;
    cilqr_joint_planner.Init(&cfg);
    if (cilqr_joint_planner.Update(&frame) != Status::SUCCESS)
    {
        std::cerr << "Update failed" << std::endl;
        return false;
    }
    // TODO(ZBW): add v and yaw to ref_line and add corresponding weight in cilqr_config if the model outputs v and yaw
    Trajectory cilqr_res = ref_line;
    cilqr_joint_planner.RunCilqrPlanner(&cilqr_res);
    return true;
}
void debugsmooth(std::vector<Traj> &modelpts)
{
    int count_failed = 0;
    for (int i = 0; i < modelpts.size(); ++i)
    {
        auto traj = modelpts[i];
        if (RoughSmooth(traj) && testsmooth(traj))
        {
        }
        else
        {
            count_failed++;
            AP_LERROR() << i << ",failed!" << ",end pt:" << traj.back().x << "," << traj.back().y;
        }
    }
    AP_LERROR() << "========================================";
    AP_LERROR() << "TOTAL FAILED:" << count_failed << ",trajs num:" << modelpts.size();
    AP_LERROR() << "========================================";
}
void debugcilqr(std::vector<Traj> &modelpts)
{
    int count_failed = 0;
    for (int i = 0; i < modelpts.size() - 1; ++i)
    {
        auto traj = modelpts[i];
        if (testcilqr(traj))
        {
        }
        else
        {
            count_failed++;
            AP_LERROR() << i << ",failed!" << ",end pt:" << traj.back().x << "," << traj.back().y;
        }
    }
    AP_LERROR() << "========================================";
    AP_LERROR() << "TOTAL FAILED:" << count_failed << ",trajs num:" << modelpts.size() - 1;
    AP_LERROR() << "========================================";
}
int main(int argc, char **argv)
{
    std::string filepath = "/home/SENSETIME/weiyimin/ws/data/pathplan/postproc/vd-planningpp-2024-08-28-16-53-40.log";
    std::vector<Traj> modelpts;
    getModelPts(filepath, modelpts);
    AP_LERROR() << "model traj size:" << modelpts.size();
    // debugsmooth(modelpts);
    debugcilqr(modelpts);
}