


from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.collections import PathCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib import colors
import numpy as np

from csts_agent.ref_path import Ref_path
from csts_msgs.msg import perception_prediction, prediction_traj, object_prediction, ego_state, map_lanes, lane

class SimVisualize(object):
    def __init__(self, windows_name="simulation", plt_name="simulation_a") -> None:
        self.windows_name = windows_name
        self.plt_name = plt_name
        self.fig = plt.figure(figsize=(15, 6))
        
        self._init_data_dicts()
        self.set_params_flag = False
        self.axes_num = 3
        if self.axes_num == 5:
            self._init_plt_5axes()
        elif self.axes_num == 3:
            self._init_plt_3axes()
        else:
            raise ValueError("axes_num should be 3 or 5")
        plt.show()
    
    def _init_data_dicts(self, ):
        self.lanes_dict: Dict[str, Tuple] = {}
        
        self.ego_path_lines:List[PathCollection] = []
        self.ego_arrow = None
        self.ego_patch = None
        self.ego_text = None
        self.ego_h_tva_dict: Dict[int, Tuple(float, float)] = {}
        
        
        self.obj_arrow_dict = {}
        self.obj_patch_dict = {}
        self.obj_text_dict = {}
        self.obj_h_tva_dict: Dict[str, Dict[int, Tuple(float, float)]] = {}
        self.obj_pred_paths_dict: Dict[str, List[PathCollection]]= {}
        
        
        
    def set_params(self, draw_arrow=True, draw_id_text=True, draw_pred_path=True, remove_obj_list:List[str]=[], draw_va_obj_ids:List[str]=[]):
        self.draw_arrow = draw_arrow
        self.draw_id_text = draw_id_text
        self.draw_pred_path = draw_pred_path
        self.remove_obj_list = remove_obj_list
        self.draw_va_obj_ids=draw_va_obj_ids
        self.text_id_fontsize = 8
        self.DisplayDISX = 40        
        self.DisplayDISY = 15
        self.set_params_flag = True
        pass
    
    def _init_plt_5axes(self, ):
        self.gs = GridSpec(90, 160)
        self.ax1 = self.fig.add_subplot(self.gs[:, :90])
        self.ax2 = self.fig.add_subplot(self.gs[0:40, 95:125])
        self.ax3 = self.fig.add_subplot(self.gs[50:, 95:125])
        self.ax4 = self.fig.add_subplot(self.gs[0:40, 130:])
        self.ax5 = self.fig.add_subplot(self.gs[50:, 130:])

        self.fig.text(0.65, 0.91, 'History', ha='center',
                      fontsize=18, color='black')
        self.fig.text(0.83, 0.91, 'Planning', ha='center',
                      fontsize=18, color='black')

        self.fig.canvas.manager.set_window_title(self.windows_name)
        self.ax1.patch.set_facecolor('lightgrey')
        self.fig.suptitle(self.plt_name)
        plt.ion()
    
    def _init_plt_3axes(self, ):
        self.gs = GridSpec(90, 240)
        self.ax1 = self.fig.add_subplot(self.gs[:, :90])
        self.ax2 = self.fig.add_subplot(self.gs[:, 110:170])
        self.ax3 = self.fig.add_subplot(self.gs[:, 180:240])

        self.fig.text(0.58, 0.91, 'Speed', ha='center',
                      fontsize=13, color='black')
        self.fig.text(0.8, 0.91, 'Accel', ha='center',
                      fontsize=13, color='black')

        self.fig.canvas.manager.set_window_title(self.windows_name)
        self.ax1.patch.set_facecolor('lightgrey')
        self.fig.suptitle(self.plt_name)
        plt.ion()
    
    def draw_lanes(self, map_lanes: map_lanes, draw_edge = True):
        # TODO: Debug
        for id, lanes in self.lanes_dict:
            for lane_ in lanes:
                lane_.pop(0).remove()
        
        ds = 0.5
        # TODO: Dynamic width
        lane_width = 3.0
        # TODO: Line type
        centor_line_type_dict = dict(color='y', linewidth=1, linestyle = '--', zorder=10)
        edge_line_type_dict = dict(color='g', linewidth=2, linestyle = '-', zorder=10)
          
        # for id, lane in map_lanes.items():
        for lane_msg in map_lanes.lanes:
            lane_msg:lane
            lane_msg.waypoints
            pts = [(pt.x, pt.y) for pt in lane_msg.waypoints]
            lane_class = Ref_path(0, pts)
            
            lane_lines_plot = []
            lane_pts = lane_class.get_pts(0, lane_class.max_s, 0.5)
            left_edge = []
            right_edge = []
            lane_cl_plot = self.ax1.plot(*zip(*lane_pts), **centor_line_type_dict)
            lane_lines_plot.append(lane_cl_plot)
            if draw_edge:
                for s in np.arange(0, lane_class.max_s, ds*2):
                    left_edge.append(lane_class.coord_lane_to_world((s, lane_width/2)))
                    right_edge.append(lane_class.coord_lane_to_world((s, -lane_width/2)))
                
                lane_ll_plot = self.ax1.plot(*zip(*left_edge), **edge_line_type_dict)
                lane_rl_plot = self.ax1.plot(*zip(*right_edge), **edge_line_type_dict)
                lane_lines_plot.append(lane_ll_plot)
                lane_lines_plot.append(lane_rl_plot)
            self.lanes_dict[str(lane_msg.lane_id)] = lane_lines_plot
    
    def draw_ego(self, ego_state_: ego_state):
        if self.ego_patch:
            self.ego_patch.remove()
        ego_color = 'green'
        x, y, yaw, length, width = ego_state_.world_coord.x, ego_state_.world_coord.y, ego_state_.world_coord.z, ego_state_.ego_box.x, ego_state_.ego_box.y
        rect = Polygon(self.polygon_xy_from_object(x, y, yaw, length, width),
                                          closed=True, facecolor=ego_color, zorder=40)
        
        # TODO: Replace Polygon with Rectangle, update xyyaw
        # rect = Rectangle((x - length / 2., y - width / 2.), length, width, angle=yaw*180/np.pi, color=color, zorder=40)
        self.ax1.add_patch(rect)
        self.ego_patch = rect
        
        if self.draw_id_text:
            if self.ego_text:
                self.ego_text.remove()
            self.ego_text = self.ax1.text(ego_state_.world_coord.x, ego_state_.world_coord.y,
                                              str(0), horizontalalignment='center', fontsize=self.text_id_fontsize, zorder=30)
        if self.draw_arrow:
            if self.ego_arrow:
                self.ego_arrow.remove()
            if ego_state_.ego_speed.z < 1:
                return
            pos = (ego_state_.world_coord.x, ego_state_.world_coord.y)
            arrow_length = ego_state_.ego_speed.z * 0.7 + 2
            arrow_point = (ego_state_.world_coord.x + arrow_length * np.cos(ego_state_.world_coord.z), ego_state_.world_coord.y + arrow_length * np.sin(ego_state_.world_coord.z))
            arrow = FancyArrowPatch(pos, arrow_point, arrowstyle='->', linewidth = 3, mutation_scale=20, color=ego_color)
            # 添加箭头对象到轴
            self.ax1.add_patch(arrow)
            self.ego_arrow = arrow
            
        # draw planning
        if self.ego_path_lines:
            for path in self.ego_path_lines:
                path.remove()
            self.ego_path_lines = []
        return
        if ego_state_.ego_planning_trajectory_tva:
            # for traj in ego_state_.planning_trajs:
            tvas = [[pt.x, pt.y, pt.z] for pt in ego_state_.ego_planning_trajectory_tva]
            xyyaw = [[pt.x, pt.y, pt.z] for pt in ego_state_.ego_planning_trajectory_xyyaw]
            path = []
            for i in range(len(tvas)):
                x,y,v = xyyaw[i][0], xyyaw[i][1], tvas[i][1]
                path.append([x,y,v])
                
            x, y, v = list(zip(*path))
            norm = colors.Normalize(vmin=0, vmax=15)
            self.ego_path_lines = [self.ax1.scatter(x, y, c = v, cmap = 'viridis', norm = norm, s = [10 for i in range(len(x))], zorder = 50)]
        
                
    
    def draw_compare_ego(self, ):
        raise NotImplementedError
    
    def draw_objs(self, ego_state: ego_state, objs_states: perception_prediction):
        keys = list(self.obj_patch_dict.keys())
        for key in keys:
            if key in self.obj_patch_dict:
                self.obj_patch_dict[key].remove()
                self.obj_patch_dict.pop(key)
            if key in self.obj_text_dict:
                self.obj_text_dict[key].remove()
                self.obj_text_dict.pop(key)
            if key in self.obj_arrow_dict:
                self.obj_arrow_dict[key].remove()
                self.obj_arrow_dict.pop(key)
            if key in self.obj_pred_paths_dict:
                for line in self.obj_pred_paths_dict[key]:
                    line.remove()
                self.obj_pred_paths_dict.pop(key)
                    
                
        self.ax1.set_xlim([ego_state.world_coord.x-self.DisplayDISX,
                          ego_state.world_coord.x+self.DisplayDISX])
        self.ax1.set_ylim([ego_state.world_coord.y-self.DisplayDISY,
                          ego_state.world_coord.y+self.DisplayDISY])
        self.ax1.set_aspect('equal')
        
        # for id, obj in objs_states.social_vehicle_states.items():
        for obj in objs_states.object_predictions:
            obj:object_prediction 
            # match obj.obj_type:
            
            # TODO: Move to params
            # if obj.obj_type == ObjType.CAR:
            #     obj_color = 'blue'
            # elif obj.obj_type == ObjType.PEDESTRAIN:
            #     obj_color = 'orange'
            # elif obj.obj_type == ObjType.CYCLIST:
            #     obj_color = 'orchid'
            # elif obj.obj_type == ObjType.OBSTACLE:
            #     obj_color = 'gray'
            # else:
            #     obj_color = 'gray'
            
            obj_color = 'blue'

            x, y, yaw, length, width = obj.world_coord.x, obj.world_coord.y, obj.world_coord.z, obj.obj_box.x, obj.obj_box.y
            rect = Polygon(self.polygon_xy_from_object(x, y, yaw, length, width),
                                            closed=True, facecolor=obj_color, zorder=20)
            self.ax1.add_patch(rect)
            self.obj_patch_dict[str(obj.obj_id)] = rect
            
            if self.draw_id_text:
                self.obj_text_dict[str(obj.obj_id)] = self.ax1.text(obj.world_coord.x, obj.world_coord.y,
                                                str(obj.obj_id), horizontalalignment='center', fontsize=self.text_id_fontsize, zorder=30)
            if self.draw_arrow:
                pos = (obj.world_coord.x, obj.world_coord.y)
                arrow_length = obj.obj_speed.z * 0.7 + 2
                arrow_point = (obj.world_coord.x + arrow_length * np.cos(obj.world_coord.z), obj.world_coord.y + arrow_length * np.sin(obj.world_coord.z))
                arrow = FancyArrowPatch(pos, arrow_point, arrowstyle='->', linewidth = 3, mutation_scale=20, color=obj_color)
                # 添加箭头对象到轴
                self.ax1.add_patch(arrow)
                self.obj_arrow_dict[str(obj.obj_id)] = arrow
                
            if self.draw_pred_path:
                pred_colors = ['b', 'y', 'r', ]
                paths = []
                for i in range(len(obj.prediction_trajs)):
                    pred: prediction_traj = obj.prediction_trajs[i]
                    path_color = pred_colors[i%len(pred_colors)]
                    
                    path = [[pt.x, pt.y] for pt in pred.predicted_traj_xyt]
                    xs, ys = list(zip(*path))
                    s = [3] * len(xs)
                    paths.append(self.ax1.scatter(xs, ys, color=path_color, zorder=20, alpha=0.8, s=s))
                self.obj_pred_paths_dict[str(obj.obj_id)] = paths
                
    def draw_history_vat(self, frame_now:int, ego_state_: ego_state, objs_states: perception_prediction):
        self.ax2.clear()
        self.ax3.clear()
        self.ego_h_tva_dict[frame_now] = [ego_state_.ego_speed.z, ego_state_.ego_acc.z]
        tvas = [[key/10, value[0], value[1]] for key, value in self.ego_h_tva_dict.items()]
        # print(tvas)
        tvas.sort(key=lambda x:x[0])
        t_s, v_s, a_s = list(zip(*tvas))
        self.ax2.plot(t_s, v_s, label='ego_v')
        self.ax2.set_ylim(0, max(v_s)+5)
        # self.ax2.set_xlabel('time')
        self.ax2.set_title('Ego Speed')
        
        self.ax3.plot(t_s, a_s, label='ego_a')
        # self.ax3.set_xlabel('time')
        self.ax3.set_ylim(-3, 3)
        self.ax3.set_title('Ego Accel')
        
        # for obj_id_str in self.draw_va_obj_ids:
        #     if obj_id_str in objs_states.social_vehicle_states.keys():
        #         obj = objs_states.social_vehicle_states[obj_id_str]
        #         value = [obj.speed.v_norm, obj.accel.a_norm]
        #         self.obj_h_tva_dict[frame_now] = value
        #     else:
        #         pass
        #     tvas = [[key/10, value[0], value[1]] for key, value in self.obj_h_tva_dict.items()]
        #     tvas.sort(key=lambda x:x[0])
        #     t_s, a_s, v_s = list(zip(*tvas))
        #     self.ax2.plot(t_s, v_s, label=obj_id_str)
            
        #     self.ax3.plot(t_s, a_s, label=obj_id_str)
        
        for obj_ in objs_states.object_predictions:
            obj_:object_prediction
            obj_id_str = str(obj_.obj_id)
            if str(obj_.obj_id) in self.draw_va_obj_ids:
                value = [obj_.obj_speed.z, obj_.obj_acc.z]
                # print(f"obj_id: {obj_.obj_id}, frame_now: {frame_now}, value: {value}")
                if obj_id_str not in self.obj_h_tva_dict:
                    self.obj_h_tva_dict[obj_id_str] = {}
                self.obj_h_tva_dict[obj_id_str][frame_now] = value
            else:
                continue
            
            tvas = [[key/10, value[0], value[1]] for key, value in self.obj_h_tva_dict[obj_id_str].items()]
            
            tvas.sort(key=lambda x:x[0])
            t_s, v_s, a_s = list(zip(*tvas))
            # print("obj_id", obj_.obj_id)
            # print(tvas)
            self.ax2.plot(t_s, v_s, label=str(obj_.obj_id))
            self.ax3.plot(t_s, a_s, label=str(obj_.obj_id))
        
        self.ax2.legend()
        self.ax3.legend()
        
    
    def draw_planning_vat(self, ego_state: ego_state, ):
        if(self.axes_num < 5):
            return
        self.ax4.clear()
        self.ax5.clear()
        max_v = 5
        for path in ego_state.planning_trajs:
            vat = [[pt.v, pt.a, pt.t] for pt in path.trajectory]
            v_s, a_s, t_s = list(zip(*vat))
            self.ax4.plot(t_s, v_s, label=path.traj_info_str)
            self.ax5.plot(t_s, a_s, label=path.traj_info_str)
            max_v = max(max_v, max(v_s))
        self.ax4.legend()
        self.ax4.set_ylim(0, max_v+5)
        self.ax4.set_title('Ego Speed')
        
        self.ax5.set_ylim(-3, 3)
        self.ax5.legend()
        self.ax5.set_title('Ego Accel')
        
            
    
    def update_plt(self, frame_now:int, ego_state_: ego_state, objs_states: perception_prediction, map_lanes: map_lanes, update_lane=True):
        if not self.set_params_flag:
            raise ValueError("set_params() is not called")
        
        if frame_now == 0:
            update_lane = True
        
        if update_lane:
            self.draw_lanes(map_lanes)
            
        
        self.draw_ego(ego_state_)
        self.draw_objs(ego_state_, objs_states)
        self.draw_history_vat(frame_now, ego_state_, objs_states)
        self.draw_planning_vat(ego_state_)
        plt.pause(0.001)
        self.fig.canvas.draw()
    
    
    def rotate_around_center(self, pts, center, yaw):
        return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


    def polygon_xy_from_object(self, x, y, yaw, length, width):
        lowleft = (x - length / 2., y - width / 2.)
        lowright = (x + length / 2., y - width / 2.)
        upright = (x + length / 2., y + width / 2.)
        upleft = (x - length / 2., y + width / 2.)
        return self.rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]), yaw=yaw)
