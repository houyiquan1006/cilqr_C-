#include "common_utils/utils.hpp"
#include <cmath>
bool readAgentFutureStatusGTFromCsv(const std::string &agent_status_filename, const std::string &agent_attrs_filename, std::vector<PredictionObject> &obstacles)
{
    PredictionObject obstacle;
    // 读取 agent attrs
    std::ifstream agent_attrs_file(agent_attrs_filename);
    if (!agent_attrs_file.is_open())
    {
        std::cerr << "File could not be opened!\n";
        return false;
    }

    // 逐行读取文件内容
    std::string agent_attrs_line;
    u_int count = 0;

    while (std::getline(agent_attrs_file, agent_attrs_line))
    {
        std::stringstream lineStream(agent_attrs_line);
        std::string cell;
        std::vector<std::string> row;

        // 以逗号为分隔读取每行的数据
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        // length, width, cls
        if (row.size() < 1) {
            std::cerr << "Incomplete data at line: " << agent_attrs_line << "\n";
            return false;
        }
        try {
            if (count == 0) {
                obstacle.length = std::stof(row[0]);
            }
            else if (count == 1) {
                obstacle.width = std::stof(row[0]);
                break;
            } 
            else{break;}
            count++;
        // 遇到陌生字符的 Error 处理
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " at line: " << agent_attrs_line << "\n";
            continue;
        
        // 数据尺寸错误的处理
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << " at line: " << agent_attrs_line << "\n";
            return false;
        }
    }
    agent_attrs_file.close();

    std::ifstream agent_status_file(agent_status_filename);
    if (!agent_status_file.is_open()) {
        std::cerr << "File could not be opened!\n";
        return false;
    }

    // 逐行读取文件内容
    std::string agent_status_line;
    std::vector<PredictionTrajectory> prediction_trajectories;
    PredictionTrajectory prediction_trajectory;
    while (std::getline(agent_status_file, agent_status_line)) {
        std::stringstream lineStream(agent_status_line);
        std::string cell; 
        std::vector<std::string> row;

        // 以逗号为分隔读取每行的数据
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        // 确保每行都有足够的数据: x, y, vx, vy, yaw, score
        if (row.size() < 6) {
            std::cerr << "Incomplete data at line: " << agent_status_line << "\n";
            return false;
        }
        PredictionTrajectoryPoint trajectory_point;
        try {
            trajectory_point.position = cv::Point2f(std::stof(row[0]), std::stof(row[1]));
            trajectory_point.direction = cv::Point2f(std::cos(std::stof(row[4])),std::sin(std::stof(row[4])));
            trajectory_point.speed = cv::Point2f(std::stof(row[2]), std::stof(row[3]));
    
            prediction_trajectory.trajectory_point_array.push_back(trajectory_point);
        
        // 遇到陌生字符的 Error 处理
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " at line: " << agent_status_line << "\n";
            continue;
        
        // 数据尺寸错误的处理
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << " at line: " << agent_status_line << "\n";
            return false;
        }
    }
    agent_status_file.close();
    prediction_trajectories.push_back(prediction_trajectory);
    obstacle.trajectory_array = prediction_trajectories;
    obstacles.push_back(obstacle);
    return true;
}

void saveTrajectoryPositionsToCSV(const Trajectory& trajectory, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    // Write header
    file << "x,y\n";

    // Write each position to the file
    for (const auto& point : trajectory.traj_point_array) {
        file << point.position.x << "," << point.position.y << "\n";
    }

    file.close();
    std::cout << "Trajectory positions saved to " << filename << std::endl;
}

bool readEgoFutureStatusGtFromCsv(const std::string& filename, double& ego_a, const double& dt){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File could not be opened!\n";
        return false;
    }

    // 逐行读取文件内容
    std::string line;
    u_int count = 0;
    double ego_v_next;
    double ego_v_next_next;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell; 
        std::vector<std::string> row;

        // 以逗号为分隔读取每行的数据
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        // 确保每行都有足够的数据: x, y, vx, vy, yaw
        if (row.size() < 5) {
            std::cerr << "Incomplete data at line: " << line << "\n";
            return false;
        }
        try {
            if (count == 0) {
                ego_v_next = std::stof(row[2]);
            }
            else if (count == 1) {
                ego_v_next_next = std::stof(row[2]);
                ego_a = (ego_v_next_next - ego_v_next) / dt;
                return true;
            } 
            else{return false;}
            count++;
        // 遇到陌生字符的 Error 处理
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " at line: " << line << "\n";
            continue;
        
        // 数据尺寸错误的处理
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << " at line: " << line << "\n";
            return false;
        }
    }
    file.close();
    return true;

}
bool readEgoCurrStatusFromCsv(const std::string& filename, 
                        double& ego_v,
                        double& ego_yaw_rate) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File could not be opened!\n";
        return false;
    }

    // 逐行读取文件内容
    std::string line;
    u_int count = 0;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell; 
        std::vector<std::string> row;

        // 以逗号为分隔读取每行的数据
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        // 确保每行都有足够的数据: x, y, heading
        if (row.size() < 1) {
            std::cerr << "Incomplete data at line: " << line << "\n";
            return false;
        }
        try {
            if (count == 0) {
                ego_v = std::stof(row[0]);
            } else if (count == 1) {
                ego_yaw_rate = std::stof(row[0]);
            }
            else{return true;}
            count++;
        // 遇到陌生字符的 Error 处理
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " at line: " << line << "\n";
            continue;
        
        // 数据尺寸错误的处理
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << " at line: " << line << "\n";
            return false;
        }
    }
    file.close();
    return true;
}
bool readReflineFromCsv(const std::string& filename, 
                        Trajectory& refline_data) {
    std::vector<TrajectoryPoint>& trajectory_points = refline_data.traj_point_array;
    // 打开指定文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File could not be opened!\n";
        return false;
    }

    // 逐行读取文件内容
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell; 
        std::vector<std::string> row;

        // 以逗号为分隔读取每行的数据
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }

        // 确保每行都有足够的数据: x, y, heading
        if (row.size() < 2) {
            std::cerr << "Incomplete data at line: " << line << "\n";
            return false;
        }
        
        // 填充 refline 数据
        TrajectoryPoint trajectory_point;
        try {
            trajectory_point.position = cv::Point2f(std::stof(row[0]), std::stof(row[1]));
    
            trajectory_points.push_back(trajectory_point);
        
        // 遇到陌生字符的 Error 处理
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " at line: " << line << "\n";
            continue;
        
        // 数据尺寸错误的处理
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << " at line: " << line << "\n";
            return false;
        }
    }

    file.close();

    // 中心差分补全 Heading 信息
    for (int i = 1; i < trajectory_points.size() - 1; i++) {
        double prev_x = trajectory_points[i - 1].position.x;
        double prev_y = trajectory_points[i - 1].position.y;

        double next_x = trajectory_points[i + 1].position.x;
        double next_y = trajectory_points[i + 1].position.y;

        double theta = std::atan2(next_y - prev_y, next_x - prev_x);

        trajectory_points[i].theta = theta;
        trajectory_points[i].direction.x = std::cos(theta);
        trajectory_points[i].direction.y = std::sin(theta);
    }
    trajectory_points[0].theta = trajectory_points[1].theta;
    trajectory_points[trajectory_points.size() - 1].theta = trajectory_points[trajectory_points.size() - 2].theta;

    trajectory_points[0].direction = trajectory_points[1].direction;
    trajectory_points[trajectory_points.size() - 1].direction = trajectory_points[trajectory_points.size() - 2].direction;

    return true;
}

// bool readBoundarySetFromFolder(const std::string& folder_name, 
//                                std::vector<std::vector<BoundaryPoint>>& boundary_set) {
//     // 确保给定的路径确实存在且为目录
//     if (!fs::exists(folder_name) || !fs::is_directory(folder_name)) {
//         std::cerr << "Given path does not exist or is not a directory: " << folder_name << std::endl;
//         return false;
//     }

//     // 遍历目录中的所有文件
//     for (const auto& entry : fs::directory_iterator(folder_name)) {
//         const auto& path = entry.path();

//         // 检查文件扩展名是否为 .csv
//         if (path.extension() == ".csv") {
//             // 读取并解析 CSV 文件，将其添加到 boundary_set 中
//             std::vector<BoundaryPoint> boundary_data;
//             if (readBoundaryFromCsv(path.string(), boundary_data)) {
//                 boundary_set.push_back(std::move(boundary_data));
//             } else {
//                 std::cerr << "Failed to read boundary data from file: " << path.string() << std::endl;
//                 return false;
//             }
//         }
//     }
//     return true;
// }

// bool readBoundaryFromCsv(const std::string& filename, 
//                          std::vector<BoundaryPoint>& boundary_data) {
//     // 打开指定文件
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "File could not be opened!\n";
//         return false;
//     }

//     // 检查文件名确定边界类型
//     BoundaryDirection boundary_type;

//     if (filename.find("left") != std::string::npos) {
//         boundary_type = BoundaryDirection::LEFT;
//     } else if (filename.find("right") != std::string::npos) {
//         boundary_type = BoundaryDirection::RIGHT;
//     } else {
//         std::cerr << "Boundary type not specified in the file name!\n";
//         file.close();
//         return false;
//     }

//     // 逐行读取文件内容
//     std::string line;
//     while (std::getline(file, line)) {
//         std::stringstream lineStream(line);
//         std::string cell; 
//         std::vector<std::string> row;

//         // 以逗号为分隔读取每行的数据
//         while (std::getline(lineStream, cell, ',')) {
//             row.push_back(cell);
//         }

//         // 确保每行都有足够的数据: x, y, heading
//         if (row.size() < 3) {
//             std::cerr << "Incomplete data at line: " << line << "\n";
//             return false;
//         }
        
//         // 填充 boundary_data 数据
//         BoundaryPoint boundary_point;
//         try {
//             boundary_point.pos = Eigen::Vector2d(std::stof(row[0]), std::stof(row[1]));
            
//             // double heading = std::stof(row[2]);
//             // boundary_point.dir = Eigen::Vector2d(std::cos(heading), std::sin(heading));
//             boundary_point.boundary_type = boundary_type;
    
//             boundary_data.push_back(boundary_point);
        
//         // 遇到陌生字符的 Error 处理
//         } catch (const std::invalid_argument& e) {
//             std::cerr << "Invalid argument: " << e.what() << " at line: " << line << "\n";
//             continue;
        
//         // 数据尺寸错误的处理
//         } catch (const std::out_of_range& e) {
//             std::cerr << "Out of range: " << e.what() << " at line: " << line << "\n";
//             return false;
//         }
//     }

//     file.close();

//     // 中心差分补全 Heading 信息
//     for (int i = 1; i < boundary_data.size() - 1; i++) {
//         double prev_x = boundary_data[i - 1].pos.x();
//         double prev_y = boundary_data[i - 1].pos.y();

//         double next_x = boundary_data[i + 1].pos.x();
//         double next_y = boundary_data[i + 1].pos.y();

//         double theta = std::atan2(next_y - prev_y, next_x - prev_x);

//         boundary_data[i].dir.x() = std::cos(theta);
//         boundary_data[i].dir.y() = std::sin(theta);
//     }
//     boundary_data[0].dir = boundary_data[1].dir;
//     boundary_data[boundary_data.size() - 1].dir = boundary_data[boundary_data.size() - 2].dir;

//     // 如果 boundary_data 的尺寸大于 20，则抽稀到 20 个点
//     if (boundary_data.size() > 20) {
//         std::vector<BoundaryPoint> thinned_boundary_data;
//         size_t total_points = boundary_data.size();
        
//         // 通过选择等间距的点来抽稀
//         for (int i = 0; i < 20; ++i) {
//             size_t index = static_cast<size_t>((static_cast<double>(i) / (20 - 1)) * (total_points - 1));
//             thinned_boundary_data.push_back(boundary_data[index]);
//         }
        
//         boundary_data = std::move(thinned_boundary_data);
//     }

//     return true;
// }

// void plotBoundaries(const std::vector<std::vector<BoundaryPoint>>& boundary_set) {
//     for (const auto& boundary : boundary_set) {
//         std::vector<double> boundary_x;
//         std::vector<double> boundary_y;
//         std::string boundary_color = (boundary[0].boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

//         for (const auto& boundary_point : boundary) {
//             boundary_x.push_back(boundary_point.pos.x());
//             boundary_y.push_back(boundary_point.pos.y());
//         }

//         plt::plot(boundary_x, 
//                   boundary_y, 
//                   {{"color", boundary_color}});
//     }
// }

// void scatterEgo(const Eigen::MatrixXd& X_seq) {
//     for (int i = 0; i < X_seq.cols(); i++) {
//         Eigen::MatrixXd X = X_seq.col(i);
//         plt::scatter(
//             std::vector<double>{X(0, 0)},
//             std::vector<double>{X(1, 0)},
//             30,
//             {{"color", "black"}}
//         );
//     }

//     plt::axis("equal");
//     plt::grid(true);
//     plt::show();
// }

// void scatterBoundaryPoint(const BoundaryPoint& boundary_point) {
//     std::string color = 
//         (boundary_point.boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

//     plt::scatter(
//         std::vector<double>{boundary_point.pos.x()},
//         std::vector<double>{boundary_point.pos.y()},
//         30,
//         {{"color", color}}
//     );
// }

// void scatterBoundaryPoint(const Eigen::MatrixXd& pos,
//                           BoundaryDirection type) {
//     std::string color = 
//         (type == BoundaryDirection::LEFT) ? "blue" : "red";

//     plt::scatter(
//         std::vector<double>{pos(0, 0)},
//         std::vector<double>{pos(1, 0)},
//         30,
//         {{"color", color}}
//     );    
// }

// void plotBoundaryPointQuiver(const BoundaryPoint& boundary_point) {
//     std::string color = 
//         (boundary_point.boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

//     plt::quiver(
//         std::vector<double>{boundary_point.pos.x()},
//         std::vector<double>{boundary_point.pos.y()}, 
//         std::vector<double>{boundary_point.dir.x()},
//         std::vector<double>{boundary_point.dir.y()},
//         {{"color", color}}               
//     ); 
// }

// void plotBoundaryPointQuiver(const Eigen::MatrixXd& pos,
//                              BoundaryPoint boundary_point) {
//     std::string color = 
//         (boundary_point.boundary_type == BoundaryDirection::LEFT) ? "blue" : "red";

//     plt::quiver(
//         std::vector<double>{pos(0, 0)},
//         std::vector<double>{pos(1, 0)}, 
//         std::vector<double>{boundary_point.dir.x()},
//         std::vector<double>{boundary_point.dir.y()},
//         {{"color", color}}               
//     );         
// }

// void plotBoundaryPointQuiver(const Eigen::MatrixXd& pos,
//                              const Eigen::MatrixXd& vector, 
//                              BoundaryDirection type) {
//     std::string color = 
//         (type == BoundaryDirection::LEFT) ? "blue" : "red";

//     plt::quiver(
//         std::vector<double>{pos(0, 0)},
//         std::vector<double>{pos(1, 0)}, 
//         std::vector<double>{vector(0, 0)},
//         std::vector<double>{vector(1, 0)},
//         {{"color", color}}               
//     );         
// }

// void plotEgoQuiver(const Eigen::MatrixXd& pos,
//                    const Eigen::MatrixXd& vector) {
//     plt::quiver(
//         std::vector<double>{pos(0, 0)},
//         std::vector<double>{pos(1, 0)}, 
//         std::vector<double>{vector(0, 0)},
//         std::vector<double>{vector(1, 0)},
//         {{"color", "black"}}             
//     );         
// }

// void plotEgoQuiver(const Eigen::MatrixXd& pos,
//                    const Eigen::MatrixXd& vector,
//                    const std::string& color) {
//     plt::quiver(
//         std::vector<double>{pos(0, 0)},
//         std::vector<double>{pos(1, 0)}, 
//         std::vector<double>{vector(0, 0)},
//         std::vector<double>{vector(1, 0)},
//         {{"color", color}}             
//     );         
// }