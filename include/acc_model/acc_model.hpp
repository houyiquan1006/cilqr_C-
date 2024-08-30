#pragma once
#include <iostream>
#include <algorithm>
#include <cmath>

class AccParam {
 public:
    AccParam() {
        // Constraints
        max_v_ = 120 / 3.6; // maximum acceptable velocity
        min_v_ = 0;         // minimum acceptable jerk
        max_accel_ = 1.0;   // maximum acceptable acceleration
        max_decel_ = 2.0;   // maximum acceptable deceleration
        max_jerk_ = 0.5;    // maximum acceptable jerk

        // IDM param
        desired_speed_ = 120 / 3.6; // target cruise speed
        reaction_time_ = 2.0;       // safety time gap
        acceleration_ = 1.5;        // maximum acceleration
        deceleration_ = 2.0;        // maximum comfort deceleration
        minimum_dist_ = 3.5;        // minimum safe distance
        free_accel_coeff_ = 4;      // free accel gain delta
        safe_braking_coeff_ = 0.99;  // proportion of CAH accel
    };

    ~AccParam(){};

    // Getter method
    double max_v() const { return max_v_; }
    double min_v() const { return min_v_; }
    double max_accel() const { return max_accel_; }
    double max_decel() const { return max_decel_; }
    double max_jerk() const { return max_jerk_; }

    double desired_speed() const { return desired_speed_; }
    double reaction_time() const { return reaction_time_; }
    double acceleration() const { return acceleration_; }
    double deceleration() const { return deceleration_; }
    double minimum_dist() const { return minimum_dist_; }
    double free_accel_coeff() const { return free_accel_coeff_; }
    double safe_braking_coeff() const { return safe_braking_coeff_; }

    // Setter method
    void setMaxV(double new_max_v) { max_v_ = new_max_v; }
    void setDesiredV(double new_desired_v) { desired_speed_ = new_desired_v; }
    void setMaxJerk(double new_max_jerk) { max_jerk_ = new_max_jerk; }

    // Reset to default
    void resetMaxV() { max_v_ = 33; }
    void resetDesiredV() { desired_speed_ = 120 / 3.6; }
    void resetMaxJerk() { max_jerk_ = 0.5; }

 private:
    // Constraints
    double max_v_;
    double min_v_;
    double max_accel_;
    double max_decel_;
    double max_jerk_;

    // IDM param
    double desired_speed_;
    double reaction_time_;
    double acceleration_;
    double deceleration_;
    double minimum_dist_;
    double free_accel_coeff_;
    double safe_braking_coeff_;
};

class AccModel {
 public:
    /**
     * @brief Initialize ACC vehicle model
     */
    AccModel(const double init_v, const double init_a, AccParam acc_param)
        : s_ego_(0), v_ego_(init_v), a_ego_(init_a), acc_param_(acc_param){};

    /**
     * @brief Destroy the Acc Model object
     */
    ~AccModel(){};

    /**
     * @brief Reset vehicle state
     */
    void reset(const double init_s, const double init_v, const double init_a) {
        s_ego_ = init_s;
        v_ego_ = init_v;
        a_ego_ = init_a;

        acc_param_.resetMaxV();
        acc_param_.resetDesiredV();
    }

    /**
     * @brief Calculate free acceleration to cruise speed
     * 
     * @return double acceleration
     */
    double calcFreeAcc() {
        const double a  = acc_param_.acceleration();
        const double b  = acc_param_.deceleration();
        const double v0 = acc_param_.desired_speed();
        const double delta = acc_param_.free_accel_coeff();

        double a_free = 0;
        if (v_ego_ < v0) {
            a_free = a * (1 - pow(v_ego_ / v0, delta));
        } else {
            a_free = -b * (1 - pow(v0 / v_ego_, a * delta / b));
        }

        return std::max(-acc_param_.max_decel(), 
                        std::min(a_free, acc_param_.max_accel()));
    }

    /**
     * @brief Calculate equilibrium criteria
     * 
     * @return double 
     */
    double calcEquilibrium(double s_obs, double v_obs) {
        const double a = acc_param_.acceleration();
        const double b = acc_param_.deceleration();
        const double T = acc_param_.reaction_time();
        const double s0 = acc_param_.minimum_dist();
        const double delta_v = v_ego_ - v_obs;
        
        double s = s_obs - s_ego_;
        if (s <= 0) {
            s = 0.1;
        }
            
        const double s_desire = s0 + std::max(
            v_ego_ * T, 
            v_ego_ * T + (v_ego_ * delta_v / (2 * sqrt(a * b)))
        );

        const double z = s_desire / s;
        return z;
    }

    /**
     * @brief Calculate constant accleration heuristic 
     * 
     * @return double 
     */
    double calcCahAcc(double s_obs, double v_obs, double a_obs) {
        const double a = acc_param_.acceleration();
        const double delta_v = v_ego_ - v_obs;
        double s = s_obs - s_ego_;
        if (s <= 0) {
            s = 0.1;
        }

        double accel = 0;
        const double a_tilde = std::min(a_obs, a);
        if (v_obs * delta_v <= -2 * s * a_tilde) {
            accel = v_ego_ * v_ego_ * a_tilde / (v_obs * v_obs - 2 * a_tilde * s);
        } else {
            const double Theta = delta_v > 0 ? 1 : 0;
            accel = a_tilde - delta_v * Theta / (2 * s);
        }

        return accel;
    }

    /**
     * @brief Calculate upper bound and lower bound of jerk
     */
    void getControlConstraints(double v_max,
                               double v_min,
                               double accel_max,
                               double decel_max,
                               double jerk_max,
                               double dt,
                               double *jerk_upper_bound,
                               double *jerk_lower_bound) {
        if (v_max < v_min || accel_max < 0 || decel_max < 0 || jerk_max < 0) {
            *jerk_upper_bound = 0;
            *jerk_lower_bound = 0;
        }

        // Calculate the maximum and minimum jerk based on velocity constraints
        const double max_jerk_v = 2 * (v_max - v_ego_ - a_ego_ * dt) / dt / dt;
        const double min_jerk_v = 2 * (v_min - v_ego_ - a_ego_ * dt) / dt / dt;

        // Calculate the maximum and minimum jerk based on acceleration
        // constraints
        const double max_jerk_a = (accel_max - a_ego_) / dt;
        const double min_jerk_a = (-decel_max - a_ego_) / dt;

        // Find the maximum allowable jerk
        *jerk_upper_bound =
            std::min(std::min(max_jerk_v, max_jerk_a), fabs(jerk_max));

        // Find the minimum allowable jerk (should be the max of the minimums)
        *jerk_lower_bound =
            std::max(std::max(min_jerk_v, min_jerk_a), -fabs(jerk_max));
    }

    /**
     * @brief Calculate ACC acceleration using IDM-CAH model
     */
    double calcIdmAcceleration(double s_obs,
                               double v_obs,
                               double a_obs) {
        const double a = acc_param_.acceleration();
        const double b = acc_param_.deceleration();
        const double v0 = acc_param_.desired_speed();
        const double T = acc_param_.reaction_time();
        const double s0 = acc_param_.minimum_dist();
        const double c = acc_param_.safe_braking_coeff();
        const double z = calcEquilibrium(s_obs, v_obs);
        const double a_free = calcFreeAcc();
        
        double IDM_accel = 0;
        if (v_ego_ < v0) {
            if (z >= 1) {
                IDM_accel = a * (1 - z * z);
            } else {
                IDM_accel = a_free * (1 - std::pow(z, 2 * a / a_free));
            }
        } else {
            if (z >= 1) {
                IDM_accel = a_free + a * (1 - z * z);
            } else {
                IDM_accel = a_free;
            }
        }
        
        double accel = 0;
        const double CAH_accel = calcCahAcc(s_obs, v_obs, a_obs);

        if (IDM_accel >= CAH_accel) {
            accel = IDM_accel;
        } else {
            const double empirical_s = s_obs - ((v_ego_ * v_ego_ - v_obs * v_obs) / 2 / acc_param_.deceleration() + v_ego_ * acc_param_.reaction_time() + acc_param_.minimum_dist());
            std::cout << "empirical_s = " << empirical_s << std::endl;
            const double empirical_equilibrium = s_obs - (v_ego_ * acc_param_.reaction_time() + acc_param_.minimum_dist());
            if (empirical_s > empirical_equilibrium) {
                std::cout << "Go Gentle!" << std::endl;
                accel = 
                    (1 - c) * IDM_accel +
                    c * (CAH_accel + 0.5 * (b + a) * std::tanh((IDM_accel - CAH_accel) / b));
            } else {
                std::cout << "Go Wild!" << std::endl;
                accel =
                    (1 - c) * IDM_accel +
                    c * (CAH_accel + b * std::tanh((IDM_accel - CAH_accel) / b));
            }

            const double empirical_dangerous_criteria = s_obs - ((v_ego_ * v_ego_ - v_obs * v_obs) / 2 / acc_param_.max_decel() + v_ego_ * acc_param_.reaction_time() + acc_param_.minimum_dist());
            if (empirical_s <= empirical_equilibrium) {
                std::cout << "Dangerous!" << std::endl;
                acc_param_.setMaxJerk(2.0);
            }
        }
        
        return std::max(-acc_param_.max_decel(), 
                        std::min(accel, acc_param_.max_accel()));
    }

    /**
     * @brief Calculate control input using IDM and CAH
     *
     * @return double
     */
    double updateControlWithObs(double s_obs,
                                double v_obs,
                                double a_obs,
                                double v_ref,
                                double dt,
                                double *dist_to_obs) {
        if (dt <= 0) {
            std::cout << "Invalid dt, dt = " << dt << std::endl;
            return 0;
        }

        // Calculate acceleration input
        acc_param_.setDesiredV(v_ref);
        const double accel = calcIdmAcceleration(s_obs, v_obs, a_obs);
        *dist_to_obs = s_obs - s_ego_;

        double jerk_upper_bound;
        double jerk_lower_bound;

        this->getControlConstraints(
            acc_param_.max_v(), 
            acc_param_.min_v(),
            acc_param_.max_accel(),
            acc_param_.max_decel(),
            acc_param_.max_jerk(),
            dt, 
            &jerk_upper_bound, 
            &jerk_lower_bound);

        const double acc_jerk = (accel - a_ego_) / dt;
        const double applied_jerk =
            std::max(jerk_lower_bound, std::min(jerk_upper_bound, acc_jerk));
        acc_param_.resetMaxJerk();

        std::cout 
            << "ACC: " 
            << "s_obs = " << s_obs 
            << ", v_obs = " << v_obs
            << ", s_ego = " << s_ego_ 
            << ", min s_desire = " << v_ego_ * acc_param_.reaction_time() + acc_param_.minimum_dist()
            << ", actual_s = " << s_obs - s_ego_
            << ", max_v = " << acc_param_.max_v()
            << ", min_jerk = " << jerk_lower_bound
            << ", max_jerk = " << jerk_upper_bound
            << ", applied_accel = " << accel
            << ", applied_jerk = " << applied_jerk
            << std::endl;;

        return applied_jerk;
    }

    /**
     * @brief Compute control input to get target speed
     */
    double updateControlWithTargetSpeed(double v_ref, double dt) {
        if (dt <= 0) {
            std::cout << "Invalid dt, dt = " << dt << std::endl;
            return 0;
        }
        
        acc_param_.setDesiredV(v_ref);
        const double accel = calcFreeAcc();

        double jerk_upper_bound;
        double jerk_lower_bound;
        this->getControlConstraints(
            acc_param_.max_v(), 
            acc_param_.min_v(),
            acc_param_.max_accel(),
            acc_param_.max_decel(),
            acc_param_.max_jerk(), 
            dt,
            &jerk_upper_bound,
            &jerk_lower_bound);

        const double acc_jerk = (accel - a_ego_) / dt;
        const double applied_jerk =
            std::max(jerk_lower_bound, 
                     std::min(jerk_upper_bound, acc_jerk));

        return applied_jerk;
    }

    /**
     * @brief Update the state based on the equations of motion
     */
    void forward(double u_jerk,
                 double dt,
                 double *s_next,
                 double *v_next,
                 double *a_next) {
        s_ego_ +=
            v_ego_ * dt + 0.5 * a_ego_ * dt * dt + u_jerk * dt * dt * dt / 6;
        v_ego_ += a_ego_ * dt + 0.5 * u_jerk * dt * dt;
        a_ego_ += u_jerk * dt;

        *s_next = s_ego_;
        *v_next = v_ego_;
        *a_next = a_ego_;
    }

    /**
     * @brief Update v_max constraints
     */
    void updateConstraintsV(double max_v) { acc_param_.setMaxV(max_v); }

    /**
     * @brief Reset v_max constraints
     */
    void restConstraintsV() { acc_param_.resetMaxV(); }

 private:
    double s_ego_;
    double v_ego_;
    double a_ego_;
    AccParam acc_param_;
};
