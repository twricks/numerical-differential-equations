#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using namespace Eigen;

// Target position as a 3D vector that updates, initially with a preset motion, then based on pursuer positions.
Vector3d target_position(double t,
                         const Vector3d& prev_target,
                         const Vector3d& prev_velocity,
                         const Vector3d& pursuer1,
                         const Vector3d& pursuer2,
                         bool accelerate,
                         double dt) {

    Vector3d velocity = prev_velocity;

    if (accelerate) {
        Vector3d d1 = pursuer1 - prev_target;
        Vector3d d2 = pursuer2 - prev_target;

        // we define a vector orthogonal to the pursuer vectors
        Vector3d ortho = d1.cross(d2);

        if (ortho.norm() > 1e-8) {
            ortho.normalize();

            // We maintain the direction of the orthogonal vector to avoid 'flipping'
            if (prev_velocity.norm() > 1e-8) {
                if (ortho.dot(prev_velocity) < 0) {
                    ortho = -ortho;
                }
            }

            
            double speed = prev_velocity.norm();
            if (speed < 1e-6) speed = 10.0;

            double target_speed = speed * 1.02;
            Vector3d desired_velocity = ortho * target_speed;

            // We limit the target's turn rate, implementing realism as in the case of an animal or a vehicle; the pursuers are not thus limited.
            double max_turn_rate = 15.0 * M_PI / 180.0;
            double max_turn = max_turn_rate * dt;

            double angle = std::acos(std::clamp(
                prev_velocity.normalized().dot(desired_velocity.normalized()),
                -1.0, 1.0));

            if (angle > max_turn) {
                Vector3d axis = prev_velocity.cross(desired_velocity);
                if (axis.norm() > 1e-8) axis.normalize();
                AngleAxisd rot(max_turn, axis);
                desired_velocity = rot * prev_velocity;
            }

            // limiting the magnitude of acceleration possible for the target when evading.
            Vector3d delta_v = desired_velocity - prev_velocity;
            double max_accel = 2.0;
            double max_delta = max_accel * dt;

            if (delta_v.norm() > max_delta) {
                delta_v = delta_v.normalized() * max_delta;
            }

            velocity = prev_velocity + delta_v;

        } else {
            // moving smoothly in the absence of nearby pursuers
            double speed = prev_velocity.norm();
            velocity = prev_velocity.normalized() * (speed + 0.5 * dt);
        }
    }

    return prev_target + velocity * dt;
}


Vector3d pursuit_rhs(const Vector3d& pursuer,
                     const Vector3d& target,
                     double v) {
    Vector3d dir = target - pursuer;
    double dist = dir.norm();
    if (dist < 1e-8) return Vector3d(0,0,0);
    return v * dir.normalized();
}

// Implementing Runge-kutta ODE solver
Vector3d rk4_step(const Vector3d& x,
                  double dt,
                  double v,
                  const Vector3d& target,
                  const Vector3d& target_velocity) {

    Vector3d k1 = pursuit_rhs(x, target, v);
    Vector3d k2 = pursuit_rhs(x + 0.5 * dt * k1,
                              target + 0.5 * dt * target_velocity, v);
    Vector3d k3 = pursuit_rhs(x + 0.5 * dt * k2,
                              target + 0.5 * dt * target_velocity, v);
    Vector3d k4 = pursuit_rhs(x + dt * k3,
                              target + dt * target_velocity, v);

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
}

int main() {
    // Initial position vectors in 3D space.
    Vector3d pursuer1(1000.0, 50.0, 50.0);
    Vector3d pursuer2(-900.0, 50.0, 50.0);
    Vector3d target(0, 0, 0);
    Vector3d target_velocity(7, 1, 0);

    double dt = 0.01;
    double t = 0.0;
    int max_iters = 5000;
    //defining the distance at which a pursuer has 'caught' the target.
    double catch_distance = 1.0;

    double base_v = 23;
    double amp = 5.0;
    double freq = 0.05;
    double phase1 = 0.7;
    double phase2 = 2.0;

    double pursuer2_boost_threshold = 500.0;
    double pursuer2_boost_factor = 1.4;

    std::ofstream file("trajectory.csv");
    file << "t,tx,ty,tz,p1x,p1y,p1z,p2x,p2y,p2z,min_distance,aware\n";

    for (int i = 0; i < max_iters; ++i) {
        //A boolean variable decides if the target/prey is evading the pursuers or not; this is set to 300 catch lengths. The boolean variable is also implemented in the output
        //movie/gif, displaying the text 'evading' when awareness has come into effect
        bool aware = (target - pursuer1).norm() < 300 * catch_distance ||
                     (target - pursuer2).norm() < 300 * catch_distance;

        //updating the target position
        Vector3d new_target = target_position(
            t, target, target_velocity,
            pursuer1, pursuer2,
            aware, dt
        );

        target_velocity = (new_target - target) / dt;
        target = new_target;

        //updating pursuer speeds
        double v1 = base_v + amp * sin(2 * M_PI * freq * t + phase1);
        double v2 = base_v + amp * sin(2 * M_PI * freq * t + phase2);

        //Because pursuer2 is in a less optimal starting point, we give it a speed boost when it comes in range.
        if ((target - pursuer2).norm() < pursuer2_boost_threshold) {
            v2 *= pursuer2_boost_factor;
        }

        // --- Update pursuers ---
        pursuer1 = rk4_step(pursuer1, dt, v1, target, target_velocity);

        // Pursuer2 also aims 'off' the target, in emulation of flanking behavior. This makes it less likely it will catch it, but makes for a more interesting animation; could be
        //removed in future instances. 
        Vector3d dir_to_target = target - pursuer2;
        Vector3d up(0,0,1);
        Vector3d flank = dir_to_target.cross(up);

        if (flank.norm() > 1e-8) {
            flank = flank.normalized() * 5.0;
        }

        Vector3d aim_point = target + flank;
        pursuer2 = rk4_step(pursuer2, dt, v2, aim_point, target_velocity);

        // Calculating pursuer distances at each step. 
        double dist1 = (target - pursuer1).norm();
        double dist2 = (target - pursuer2).norm();
        double min_distance = std::min(dist1, dist2);

        file << t << ","
             << target.x() << "," << target.y() << "," << target.z() << ","
             << pursuer1.x() << "," << pursuer1.y() << "," << pursuer1.z() << ","
             << pursuer2.x() << "," << pursuer2.y() << "," << pursuer2.z() << ","
             << min_distance << ","
             << aware << "\n";

        if (dist1 < catch_distance) {
            std::cout << "Target caught by pursuer1 at t = " << t << std::endl;
            break;
        }
        if (dist2 < catch_distance) {
            std::cout << "Target caught by pursuer2 at t = " << t << std::endl;
            break;
        }

        t += dt;
    }

    file.close();
    std::cout << "Simulation complete. Output written to trajectory.csv\n";
    return 0;
}