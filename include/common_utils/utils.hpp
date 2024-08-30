#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <experimental/filesystem>

#include "common_utils/data_struct.hpp"
#include "third_party/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::experimental::filesystem;
/**
 * @brief 从 .csv 文件中读取 AgentFutureStatusGT 信息 [x , y, vx, vy, yaw, score]
 * 
 * @param agent_status_filename     [in ] 文件名
 * @param agent_attrs_filename     [in ] 文件名
 * @param obstacles [out] 障碍物列表
 * @return true  读取成功
 * @return false 读取失败
 */
bool readAgentFutureStatusGTFromCsv(const std::string& agent_status_filename, const std::string& agent_attrs_filename, std::vector<PredictionObject>& obstacles);
/**
 * @brief 将cilqr 优化后轨迹位置信息 [x , y] 导入 .csv 文件中
 * 
 * @param trajectory     [in ] cilqr 优化后轨迹
 * @param filename [out] 文件名
 * @return void
 */
void saveTrajectoryPositionsToCSV(const Trajectory& trajectory, const std::string& filename);
/**
 * @brief 从 .csv 文件中读取 EgoFutureStatusGT 信息 [x , y, vx, vy, yaw]
 * 
 * @param filename     [in ] 文件名
 * @param ego_v_next [out] 自车速度
 * @return true  读取成功
 * @return false 读取失败
 */
bool readEgoFutureStatusGtFromCsv(const std::string& filename, double& ego_v_next, const double& dt);
/**
 * @brief 从 .csv 文件中读取 EgoCurrStatus 信息 [v , yaw_rate, traffic_light, ego_light]
 * 
 * @param filename     [in ] 文件名
 * @param ego_v [out] 自车速度
 * @param ego_yaw_rate [out] 自车角速度
 * @return true  读取成功
 * @return false 读取失败
 */
bool readEgoCurrStatusFromCsv(const std::string &filename,
                              double &ego_v,
                              double &ego_yaw_rate);
/**
 * @brief 从 .csv 文件中读取 Refline 离散路径点信息
 * 
 * @param filename     [in ] 文件名
 * @param refline_data [out] 离散路径点数据
 * @return true  读取成功
 * @return false 读取失败
 */
bool readReflineFromCsv(
    const std::string& filename, 
    Trajectory& refline_data);

/**
 * @brief 读取给定的目录中所有的 CSV 文件，并添加到 boundary_set 中
 * 
 * @param folder_name  文件目录
 * @param boundary_set boundary 集合
 * @return true  读取成功
 * @return false 读取失败
 */
bool readBoundarySetFromFolder(
    const std::string& folder_name, 
    std::vector<std::vector<BoundaryPoint>>& boundary_set);

/**
 * @brief 从 .csv 文件中读取 Boundary 离散路径点信息
 * 
 * @param filename      [in ] 文件名
 * @param boundary_data [out] 离散路径点数据
 * @return true  读取成功
 * @return false 读取失败
 */
bool readBoundaryFromCsv(
    const std::string& filename, 
    std::vector<BoundaryPoint>& boundary_data);

/**
 * @brief 绘制离散点形式的边界
 * 
 * @param boundary_set 一堆边界
 */
void plotBoundaries(const std::vector<std::vector<BoundaryPoint>>& boundary_set);

/**
 * @brief 绘制自车位置
 * 
 * @param X_seq 自车位置表示为一个 2*1 的向量 [x; y]，位置序列是个 2*N 的向量
 */
void scatterEgo(const Eigen::MatrixXd& X_seq);

/**
 * @brief 绘制一个边界点
 * 
 * @param boundary_point 边界点
 */
void scatterBoundaryPoint(const BoundaryPoint& boundary_point);

/**
 * @brief 另一种形式的画 Boundary
 * 
 * @param pos  boundary 点坐标
 * @param type boundary 类型
 */
void scatterBoundaryPoint(const Eigen::MatrixXd& pos,
                          BoundaryDirection type);

/**
 * @brief 绘制边界点的切向量
 * 
 * @param boundary_point 边界点
 */
void plotBoundaryPointQuiver(const BoundaryPoint& boundary_point);

/**
 * @brief 另一种形式的画 quiver
 * 
 * @param pos            boundary 点坐标
 * @param boundary_point boundary 类型
 */
void plotBoundaryPointQuiver(const Eigen::MatrixXd& pos,
                             BoundaryPoint boundary_point);

/**
 * @brief 另一种形式的画 quiver
 * 
 * @param pos    quiver 起点
 * @param vector quiver 方向矢量
 * @param type   boundary 类型
 */
void plotBoundaryPointQuiver(const Eigen::MatrixXd& pos,
                             const Eigen::MatrixXd& vector, 
                             BoundaryDirection type);

/**
 * @brief 绘制 Normal Vector
 * 
 * @param pos    向量起点
 * @param vector 向量方向
 */
void plotEgoQuiver(const Eigen::MatrixXd& pos,
                   const Eigen::MatrixXd& vector);

/**
 * @brief 绘制最终的 Gradient
 */
void plotEgoQuiver(const Eigen::MatrixXd& pos,
                   const Eigen::MatrixXd& vector,
                   const std::string& color);