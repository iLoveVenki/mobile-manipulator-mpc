import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class Base:
    def __init__(self, dt):
        self.dt = dt
        self.base_length = 2 * (0.7/2 + 0.157) # estimated in simulation  
        self.base_width = 0.52 # estimated in simulation
          
    def base_radius(self):
        # base_radius = math.sqrt((self.base_length / 2.)**2 + (self.base_width / 2.) ** 2)
        # return base_radius
        return 0.4 # 0.57
    
    def f_kinematics(self, x, u, limited_yaw=False):
        """根据车辆的当前位置x和控制量u，更新计算车辆的下一个状态

        Args:
            x (_type_): 包含车辆位置和速度的状态向量
            u (_type_): 包含车辆加速度的控制向量
            limited_yaw (bool, optional): 是否限制车辆转角到[-π, π)范围内的flag标志. Defaults to False.
        """
        # TODO: assert x.type == u.type == casadi variable
        # 计算车辆的位移和速度
        x_next = ca.horzcat(
            x[0] + self.dt * x[3], #? 这里的模型推导看不懂，应该看一下轮式机器人的差分运动学模型
            x[1] + self.dt * x[4],
            x[2] + self.dt * x[5], 
            x[3] + self.dt * (u[0]*np.cos(x[2])- x[4]*x[5]), #- x[4] * x[5] #- x[3] * np.tan(x[2]) * x[5]),
            x[4] + self.dt * (u[0]*np.sin(x[2]) + x[3]*x[5]), #+ x[3] * x[5]),
            x[5] + self.dt * u[1] # 计算车辆的角速度和角速度变化量
        ) # x[0] = x, x[1] = y, x[2] = psi, x[3] = x_dot, x[4] = y_dot, x[5] = psi_dot, u[0] = v_dot, u[1] = w_dot

        # 如果limited_yaw参数为True，函数会将角度限制在[-π, π)范围内
        if limited_yaw:
            x_next[2] = ca.fmod((x_next[2] + ca.pi), (2*ca.pi)) - ca.pi # to [-pi, pi)
        
        return x_next