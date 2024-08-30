#include "speed_mpc_solver.hpp"
#include "ad_log.h"
namespace senseAD
{
    namespace perception
    {
        namespace camera
        {

            void getweight(SpeedOptimizeWeight &weight, std::string scenetype)
            {
                auto weights = OSQPHint::weight_config;
                if (weights.count(scenetype) == 0)
                {
                    AD_LERROR() << "undefined speed scenetype" << std::endl;
                    return;
                }
                weight.s_weight = weights[scenetype][0];
                weight.vel_weight = weights[scenetype][1];
                weight.acc_weight = weights[scenetype][2];
                weight.jerk_weight = weights[scenetype][3];
                weight.curve_weight = weights[scenetype][4];

                return;
            }
            void testspeedsolver(std::vector<std::pair<float, float>> pts)
            {
                OptimizeHorizon horizon;
                auto speed_optimizer = SpeedMpcSolver(horizon);
                SpeedOptimizeState initstate(0, 0.0, 0.0);
                SpeedOptimizeWeight weight;
                getweight(weight, "redlight");
                // assume stop at s = 30m,
                double stop_s = 30;
                std::vector<SpeedOptimizeProfile> profiles;
                int profile_size = horizon.horizon_step + 1;
                profiles.resize(profile_size);
                double tstep = 0.2;
                double expected_speed = 8.33;
                double tend = 8.0;
                double meana = (expected_speed - initstate.vel) / tend;
                // double meana = -initstate.vel * initstate.vel * 0.5 / stop_s;
                IDMParam idmparam;
                idmparam.desire_velocity = expected_speed;
                idmparam.min_spacing = initstate.vel * 1.0;
                if (idmparam.min_spacing < 1.0)
                    idmparam.min_spacing = 1.0;
                float lastv = initstate.vel;
                float lasts = initstate.s_frenet;
                double tcur = tstep;
                std::vector<float> acclist, detavlist;
                for (auto &profile : profiles)
                {
                    profile.confidence_factor = 1.0;
                    profile.s_leader = 1000.0;
                    profile.s_follower = -1000.0;
                    profile.s_upper = 1000.0;
                    profile.s_lower = -100.0;
                    profile.v_upper = 25.0;
                    profile.v_lower = 0.0;
                    profile.a_upper = 2.5;
                    profile.a_lower = -3.5;
                    profile.s_ref_collision = 0.0;
                    profile.t_seq = tcur;
                    profile.jerk_lower = -4.0;
                    profile.jerk_upper = 4.0;
                    // calculate ref sva
                    double vcur = initstate.vel + meana * tstep;
                    double scur = initstate.vel * tstep + 0.5 * meana * tstep * tstep;
                    double detas = stop_s - lasts;
                    double detav = lastv - 0.0;
                    double acc = IDMPlanner::CalcFreeAcc(idmparam, lastv);
                    // double acc = IDMPlanner::CalcIDMAcc(idmparam, detas, lastv, vcur);
                    if (tcur <= 0.2)
                        acc = 0;
                    acclist.push_back(acc);
                    AD_LERROR() << "time:" << tcur << ",acc:" << acc << std::endl;
                    profile.a_ref_cruise = acc;
                    profile.v_ref_cruise = std::max(0.0, lastv + acc * 0.2);
                    profile.s_ref_cruise =
                        lasts + (lastv + profile.v_ref_cruise) * 0.5 * 0.2;

                    //
                    tcur += 0.2;
                    lastv = profile.v_ref_cruise;
                    lasts = profile.s_ref_cruise;
                }
                speed_optimizer.Update(initstate, profiles, weight);
                std::vector<SpeedOptimizeState> states;
                std::vector<SpeedOptimizeControl> controls;

                if (!speed_optimizer.Solve())
                {
                    AD_LERROR() << "wrong!" << std::endl;
                }
                else
                {
                    speed_optimizer.GetSolution(&states, &controls);
                    std::vector<float> xlist, slist, vlist, alist;
                    float lastv = initstate.vel;
                    float lasta = 0.0;
                    float lastk = 0.0;
                    for (int i = 0; i < states.size(); ++i)
                    {
                        xlist.push_back(0.2 * i);
                        slist.push_back(states[i].s_frenet);
                        vlist.push_back(states[i].vel);
                        alist.push_back(states[i].acc);
                        detavlist.push_back(lastv + 0.2 * lasta);
                        lastv = states[i].vel;
                        lasta = states[i].acc;
                        lastk = controls[i].jerk;
                    }
                }
                return;
            }
        } // namespace camera
    } // namespace perception
} // namespace senseAD