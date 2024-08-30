#include <iostream>
#include <numeric>
#include <vector>
#include <string>

#include "acc_model/acc_model.hpp"
#include "third_party/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;

void plotCruiseData(const std::vector<double>& timeline,
                    const std::vector<double>& s_forward_list,
                    const std::vector<double>& v_forward_list,
                    const std::vector<double>& a_forward_list,
                    const std::vector<double>& jerk_forward_list,
                    bool cruise_only) {
    plt::figure_size(1500, 600);

    int fig_num = 4;
    if (cruise_only) {
        fig_num = 3;
    }

    // s-t
    plt::subplot(1, 2, 1);
    plt::plot(
        timeline, 
        s_forward_list, 
        {{"label", "forward sim result"}}
    );
    plt::xlabel("time (s)");
    plt::ylabel("accumulated length (m)");
    plt::legend();
    plt::grid(true);

    // v-t
    plt::subplot(fig_num, 2, 2);
    plt::plot(timeline, v_forward_list);
    plt::xlabel("time (s)");
    plt::ylabel("velocity (m/s)");
    plt::grid(true);

    // a-t
    plt::subplot(fig_num, 2, 4);
    plt::plot(timeline, a_forward_list);
    plt::xlabel("time (s)");
    plt::ylabel("acceleration (m/s^2)");
    plt::grid(true);
    
    // jerk-t
    plt::subplot(fig_num, 2, 6);
    plt::plot(timeline, jerk_forward_list);
    plt::xlabel("time (s)");
    plt::ylabel("jerk (m/s^3)");
    plt::grid(true);

    // Display
    if (cruise_only) {
        // display
        plt::show();  
    }
};

void plotAccData(const std::vector<double>& timeline,
                 const std::vector<double>& s_obs_list,
                 const std::vector<double>& s_emperical_list,
                 const std::vector<double>& dist_forward_list) {
    plt::subplot(1, 2, 1);
    plt::plot(
        timeline, 
        s_emperical_list, 
        {{"label", "emperical"}}
    );

    plt::plot(
        timeline, 
        s_obs_list, 
        {{"label", "predicted obs"}}
    );
    plt::xlabel("time (s)");
    plt::ylabel("accumulated length (m)");
    plt::legend();
    plt::grid(true);

    // dist-t
    plt::subplot(4, 2, 8);
    plt::plot(timeline, dist_forward_list);
    plt::xlabel("time (s)");
    plt::ylabel("collision distance (m)");
    plt::grid(true);

    // display
    plt::show();   
}

void setTimeLine(int dense_horizon,
                 int sparse_horizon,
                 double dense_timestep,
                 double sparse_timestep,
                 std::vector<double>* timeline,
                 std::vector<double>* timestep) {
    timeline->reserve(dense_horizon + sparse_horizon);
    timestep->reserve(dense_horizon + sparse_horizon);

    // Fill time steps
    for (int i = 0; i < dense_horizon; ++i) {
        timeline->push_back(dense_timestep);
        timestep->push_back(dense_timestep);
    }

    for (int i = 0; i < sparse_horizon; ++i) {
        timeline->push_back(sparse_timestep);
        timestep->push_back(sparse_timestep);
    }

    // Compute cumulative sum of timeline
    for (size_t i = 1; i < timeline->size(); ++i) {
        (*timeline)[i] += (*timeline)[i - 1];
    }
}

void setObstacle(double s_obs, 
                 double v_obs,
                 const std::vector<double> timeline,
                 std::vector<double>* s_obs_list) {
    s_obs_list->resize(timeline.size());
    for (int i = 0; i < static_cast<int>(timeline.size()); i++) {
        (*s_obs_list)[i] = s_obs + timeline[i] * v_obs;
    }
}

void testAccCarFollowing(
    double s_obs, 
    double v_obs, 
    double obs_length, 
    double s_ego, 
    double v_ego, 
    double a_ego,
    const std::vector<double>& timestep,
    const std::vector<double>& timeline) {

    std::vector<double> s_obs_list;
    std::vector<double> s_emperical_list;
    std::vector<double> s_forward_list;
    std::vector<double> v_forward_list;
    std::vector<double> a_forward_list;
    std::vector<double> jerk_forward_list;
    std::vector<double> dist_forward_list;
    
    s_emperical_list.resize(timeline.size());
    s_forward_list.resize(timeline.size());
    v_forward_list.resize(timeline.size());
    a_forward_list.resize(timeline.size());
    jerk_forward_list.resize(timeline.size());
    dist_forward_list.resize(timeline.size());    

    // Obstacle prediction info
    setObstacle(s_obs, v_obs, timeline, &s_obs_list);

    // Create ACC Planner
    AccParam acc_param;
    AccModel acc_model(s_ego, v_ego, a_ego, acc_param);

    for (int i = 0; i < static_cast<int>(timeline.size()); i++) {
        const double dt = timestep[i];
        const double s_obs = s_obs_list[i];

        double s_ref;
        double delta_s;
        const double jerk = acc_model.UpdateControlWithObs(
            s_obs, v_obs, dt, obs_length, &s_ref, &delta_s
        );

        double s_next;
        double v_next;
        double a_next;
        acc_model.Forward(jerk, dt, &s_next, &v_next, &a_next);

        s_emperical_list[i] = s_ref;
        s_forward_list[i] = s_next;
        v_forward_list[i] = v_next;
        a_forward_list[i] = a_next;
        jerk_forward_list[i] = jerk;
        dist_forward_list[i] = delta_s;
    }

    plotCruiseData(timeline,
                   s_forward_list,
                   v_forward_list,
                   a_forward_list,
                   jerk_forward_list,
                   false);   

    plotAccData(timeline,
                s_obs_list,
                s_emperical_list,
                dist_forward_list); 
}

void testCruiseControl(double v_ref,
                       double s_ego, 
                       double v_ego, 
                       double a_ego,
                       const std::vector<double>& timestep,
                       const std::vector<double>& timeline) {
    // Init vectors
    std::vector<double> s_forward_list;
    std::vector<double> v_forward_list;
    std::vector<double> a_forward_list;
    std::vector<double> jerk_forward_list;

    s_forward_list.resize(timeline.size());
    v_forward_list.resize(timeline.size());
    a_forward_list.resize(timeline.size());
    jerk_forward_list.resize(timeline.size());

    // Create ACC Planner
    AccParam acc_param;
    AccModel acc_model(s_ego, v_ego, a_ego, acc_param);

    for (int i = 0; i < static_cast<int>(timeline.size()); i++) {
        double dt = timestep[i];
        double jerk = acc_model.UpdateControlWithTargetSpeed(v_ref, dt);

        double s_next;
        double v_next;
        double a_next;
        acc_model.Forward(jerk, dt, &s_next, &v_next, &a_next);   

        s_forward_list[i] = s_next;
        v_forward_list[i] = v_next;
        a_forward_list[i] = a_next;
        jerk_forward_list[i] = jerk;
    }

    plotCruiseData(timeline,
                   s_forward_list,
                   v_forward_list,
                   a_forward_list,
                   jerk_forward_list,
                   true);   

}

int main() {
    // Planning horizon
    const int dense_horizon = 5;
    const int sparse_horizon = 5;
    const double dense_timestep = 0.2;
    const double sparse_timestep = 0.4; 

    // Calculate timeline
    std::vector<double> timeline;
    std::vector<double> timestep;

    setTimeLine(dense_horizon, 
                sparse_horizon, 
                dense_timestep, 
                sparse_timestep, 
                &timeline,
                &timestep);

    // Obstacle info
    const double s_obs = 20;
    const double v_obs = 80 / 3.6;
    const double obs_length = 4;

    // Ego info
    const double s_ego = 0;
    const double a_ego = 0;

    // Cruise speed
    const double v_ref = 20;

    for (double v_ego = 50 / 3.6; v_ego < 100 / 3.6; v_ego += 20 / 3.6) {
        // Forward Sim
        testAccCarFollowing(s_obs, v_obs, obs_length,
                            s_ego, v_ego, a_ego, 
                            timestep,
                            timeline);
    }

    std::cout << "Another One!" << std::endl;

    for (double v_ego = 50 / 3.6; v_ego < 120 / 3.6; v_ego += 20 / 3.6) {
        testCruiseControl(v_ref, s_ego, v_ego, a_ego,
                          timestep, timeline);    
        
    }

    return 0;
}
