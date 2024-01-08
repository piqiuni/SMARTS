


from collections import namedtuple
from smarts.core.road_map import RoadMap


from smarts.core.observations import EgoVehicleObservation, Observation, VehicleObservation


def get_traj_to_lane(ego: EgoVehicleObservation, lane_id : str, map : RoadMap) :
    # Pt = namedtuple("Pt", [("x", float), ("y", float), ("t", float), ("yaw", float), ("v", float), ("a", float)])
    Pt = namedtuple("Pt", ["x", "y", "t", "yaw", "v", "a"])

    traj = List[Pt]




    return traj

"









参考：

std::vector<float> calBicycleModelStep(float x, float y, float yaw, float v, float a, float delta, float dt)
    {
        std::vector<float> value;
        float f_len = 2.4;
        float r_len = 2.4;
        float L = 4.8;
        float L_wheel_base = 2.9;

        float new_x = x + v * cos(yaw) * dt;
        float new_y = y + v * sin(yaw) * dt;
        float new_v = std::max(0.0f, v + a*dt); 
        float new_yaw = yaw + v / L_wheel_base * tan(delta) * dt;
        // printf("yaw: %.1f, v: %.1f, a: %.1f, delta: %.1f\n", new_x, new_y, new_yaw, v, new_a, delta);

        new_v = v + a*dt; 
        value.emplace_back(new_x);
        value.emplace_back(new_y);
        value.emplace_back(new_yaw);
        value.emplace_back(new_v);
        value.emplace_back(a);
        return value;
    }
    
    
     float calPurePersuit(float ego_yaw, float d_angle, float L_ahead)
    {
        float L_wheel_base = 2.9; // wheelbase
        float alpha = d_angle - ego_yaw;
        float delta = alpha != 0 ? atan2(2.0 * L_wheel_base * sin(alpha) / L_ahead, 1.0) : 0;
        // printf(", alpha: %.1f°, delta: %.1f°\n", alpha*180/2/3.1415926, delta*180/2/3.1415926);
        return delta;
    }




{
                // use follow lane path
                ego_state_->lane_id;
                ego_state_->ego_speed.z * 5;
                float s_now = ego_lane_coord[0];
                float d_now = ego_lane_coord[1];
                float x_now = ego_state_->world_coord.x;
                float y_now = ego_state_->world_coord.y;
                float yaw_now = ego_state_->world_coord.z;
                float v_now = ego_state_->ego_speed.z;
                float a_now = ego_state_->ego_acc.z;
                follow_path.emplace_back(Eigen::Vector4f(x_now, y_now, yaw_now, 0.0));
                collision_check_xytyaw_list.emplace_back(Eigen::Vector4f(x_now, y_now, 0.0, yaw_now));
                float di = 1;
                for(int i = 1; i < 51; i += di)
                {
                    float time_now = i*0.1;
                    float track_dis = ego_state_->ego_speed.z * 1.5;
                    auto track_pt = ego_lane.coord_lane_to_world(Eigen::Vector3f(s_now+track_dis, 0, 0));
                    float d_angle = atan2(track_pt[1] - ego_state_->world_coord.y, track_pt[0] - ego_state_->world_coord.x);
                    float delta = calPurePersuit(ego_state_->world_coord.z, d_angle, track_dis);
                    float a_now = std::max(break_acc, a_now - max_jerk*di/10);

                    auto next_xyyawva = calBicycleModelStep(x_now, y_now, yaw_now, v_now, a_now, delta, di/10);
                    
                    x_now = next_xyyawva[0];
                    y_now = next_xyyawva[1];
                    yaw_now = next_xyyawva[2];
                    v_now = next_xyyawva[3];
                    a_now = next_xyyawva[4];
                    auto lane_coord = ego_lane.coord_world_to_lane(Eigen::Vector3f(x_now, y_now, 0));
                    s_now = lane_coord[0];
                    d_now = lane_coord[1];
                    follow_path.emplace_back(Eigen::Vector4f(x_now, y_now, yaw_now, time_now));
                    if(i % 5 == 0)
                    {
                        collision_check_xytyaw_list.emplace_back(Eigen::Vector4f(x_now, y_now, time_now, yaw_now));
                    }
                }
            }