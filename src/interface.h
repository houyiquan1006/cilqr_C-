#include "fem_pos_deviation_smoother.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include "ad_log.h"
struct modelpt
{
    /* data */
    float x = 0.0f;
    float y = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float yaw = 0.0f;
};
typedef std::vector<modelpt> Traj;

Traj parsePoints(const std::string &line)
{
    Traj points;
    std::istringstream stream(line);
    std::string token;

    while (std::getline(stream, token, ';'))
    {
        std::istringstream pointStream(token);
        modelpt pt;
        char separator;

        pointStream >> pt.x >> separator >> pt.y >> separator >> pt.vx >> separator >> pt.vy >> separator >> pt.yaw;
        points.push_back(pt);
        // std::cerr << "get pt:" << pt.x << "," << pt.y << std::endl;
    }

    return points;
}
bool getModelPts(std::string path, std::vector<Traj> &trajs)
{
    std::ifstream file(path);
    std::string line;
    trajs.clear();
    std::string flag = "model ori pts:";
    if (file.is_open())
    {
        AP_LERROR() << "START READ FILE:" << path;
        while (std::getline(file, line))
        {
            if (line.find(flag) != std::string::npos)
            {
                // 解析并处理 "model ori pts" 行
                std::string data = line.substr(line.find(flag) + flag.size());
                Traj points = parsePoints(data);
                trajs.push_back(points);
            }
        }
        file.close();
    }
    else
    {
        std::cerr << "无法打开文件: " << path << std::endl;
    }
    return true;
}