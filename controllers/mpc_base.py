import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class MPCBase:
    def __init__(self,
        robot, 
        obstacle_list,
        N = 10, # N为预测时阈的宽度
        Q = np.diag([5., 5., 0.0, 0, 0, 1.]), # x y psi dx dy dpsi 成本函数中的状态误差权重
        P = np.diag([5., 5., 0.0, 0, 0, 1.]),  # [2., 2., 0., 0, 0, 1.]
        R = np.diag([1., 1.]), # 成本函数中的控制误差权重
        M = np.diag([1e5]), # 1e3, th=0.0, r=0.4
        ulim=np.array([[-2, -ca.pi],[2, ca.pi]]), # dv, dw
        xlim=np.array([[-100, -100, -2, -2, -ca.pi],[100, 100, 2, 2, ca.pi]]) # x, y, _, dx, dy, dpsi
        ):

        self.Q_value = Q
        self.R_value = R
        self.P_value = P
        self.M_value = M
        self.dt = robot.dt
        self.N = N
        self.ulim = ulim # 基座的速度和角速度范围
        self.xlim = xlim # 运动范围
        self.f_dynamics = robot.f_kinematics #member function
        self.base_radius = robot.base_radius #member function
        self.obstacle_list = obstacle_list
        self.reset()

    def slackObsAvoid(self, obstacle_list, x):
        """
        not used!!! wrong implementation, slack variable should be explicitly used as the boundary of constraints, to avoid fmax() and fabs()
        returns the norm of the violation of constraints
        """
        expand_dist = 0.0
        sum_slack_var = 0
        for obs in obstacle_list:
            # dist_to_obs = ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2) - (obs.radius + self.base_radius())
            # sum_slack_var += ca.fabs(ca.fmin(dist_to_obs - expand_dist, 0))
            dist_to_obs = ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2) - (obs.radius + self.base_radius())
            # sum_slack_var += ca.fmax(- dist_to_obs + expand_dist, 0)
            sum_slack_var += - dist_to_obs + expand_dist
        return sum_slack_var
        #R_base+R_obstacle-|postion_base-postion_obstacle|<=0
        # return (x[0]-obstacle.x)**2 + (x[1]-obstacle.x)**2 - obstacle.radius - self.base_radius() 

    def obsAvoid(self, obstacle_list, x):
        g = [] # 安全成本，该数值应该小于0，负数的值越大，则越加安全
        threshold = 0.0 # 0.5 is safe, -0.1 is elegant
        for obs in obstacle_list:
            g.append((obs.radius + self.base_radius()) - ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2) + threshold) # should be <= 0
        # 障碍物半径+基座半径+安全距离 - 障碍物中心到基座中心的距离应该小于0
        return g # all elements should be <= 0
    
    def angleDiff(self, a, b):
        """
        input angle from any range, output a-b converted to [-pi, pi)
        in this program a and b are equivalent because we use the squared error
        """
        a = ca.fmod((a + ca.pi), (2*ca.pi)) - ca.pi # convert to [-pi.pi)
        b = ca.fmod((b + ca.pi), (2*ca.pi)) - ca.pi

        # try:
        #     if a * b >= 0: # both [-pi, 0] or [0, pi) 
        #         angle_diff = a - b
        #     elif a > b: # a (0, pi), b [-pi, 0)
        #         angle_diff = a - b if a - b <= ca.pi else a - b - 2*ca.pi
        #     elif a < b: # a [-pi, 0), b (0, pi)
        #         angle_diff = a - b if a - b > -ca.pi else a - b + 2*ca.pi
        #     print("=================angle diff working=================")
        # except:
        #     angle_diff = a - b
        #     print("angle diff not working because of casadi")
        
        angle_diff = ca.if_else(
            a * b >= 0, 
            a - b, 
            ca.if_else(
                a > b, 
                ca.if_else(
                    a - b <= ca.pi, 
                    a - b, 
                    a - b - 2 * ca.pi
                ), 
                ca.if_else(
                    a - b > -ca.pi, 
                    a - b, 
                    a - b + 2 * ca.pi
                )
            )
        )

        return angle_diff

    def setWeight(self, Q=None, R=None, P=None, M=None):
        """该方法接受 4 个可选参数（Q、R、P、M），分别对应不同的权重值。"""
        if Q is not None: 
            self.Q_value = Q
            
        if R is not None: 
            self.R_value = R
            
        if P is not None: 
            self.P_value = P
            
        if M is not None: 
            self.M_value = M
            
        self.opti.set_value(self.Q, self.Q_value)
        self.opti.set_value(self.R, self.R_value)
        self.opti.set_value(self.P, self.P_value)
        self.opti.set_value(self.M, self.M_value)

    def reset(self):
        # Define optimization variables
        self.opti = ca.Opti()

        '''
        reason for the implementation that X is Nx2 not 2xN: 
        given p is a vector of same dim of x
        if we use Nx2, then matrix mult = X @ p; and elementwise mult = X * p, 
        of we use 2xN, then matrix mult = (X.T @ p).T; and elementwise mult = (X.T * p).T
        '''
        # opti.variable()函数的主要目的是封装对 _variable 方法的调用，并在调用期间捕获异常信息以及调用堆栈信息。这样做可能有助于调试和记录有关方法调用的详细信息。
        self.X = self.opti.variable(self.N+1, 6)    # states
        self.U = self.opti.variable(self.N, 2)      # inputs
        self.s = self.opti.variable(self.N+1, 1)    # slack variable（边缘变量），用于表示优化问题的约束。

        # 初始化机器人系统的初始状态、参考状态和参考控制
        self.X_init = self.opti.parameter(1, 6)
        self.X_ref = self.opti.parameter(self.N+1, 6)
        self.U_ref = self.opti.parameter(self.N, 2)

        self.X_guess = None
        self.U_guess = None

        # 在优化问题中，我们可以使用这些权重矩阵来调整优化目标的权重，从而获得更优的解。
        self.Q = self.opti.parameter(6,6) # 成本函数中的状态误差分量权重
        self.R = self.opti.parameter(2,2) #& 成本函数中的控制误差分量权重
        self.P = self.opti.parameter(6,6) #* 成本函数中的终止状态误差分量权重
        self.M = self.opti.parameter(1,1) # 成本函数中的避碰分量权重

        self.setWeight()

        cost = 0
        # Define constraints and cost, casadi requires x[k, :] instead of x[k] (which will be shape(1,1)) when calling row vector
        for k in range(self.N): #! range()函数生成的结果是左闭右开，遍历的最后一个值为N-1，不包括N
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :])) # 第 k+1 步的状态预测
            # 状态误差是第 k 步的实际状态 self.X[k, :] 与参考状态 self.X_ref[k, :] 的差值。
            state_error = ca.horzcat(
                self.X[k, :2] - self.X_ref[k, :2],
                self.angleDiff(self.X[k, 2], self.X_ref[k, 2]),
                self.X[k, 3:] - self.X_ref[k, 3:]
            )
            # 控制误差是第 k 步的实际控制输入 self.U[k, :] 与参考控制输入 self.U_ref[k, :] 的差值。
            control_error = self.U[k, :] - self.U_ref[k, :]
            cost += ca.mtimes([state_error, self.Q, state_error.T]) \
                                + ca.mtimes([control_error, self.R, control_error.T])
            # 设置控制输入self.U[k, :]的上、下界
            self.opti.subject_to(self.opti.bounded(self.ulim[0].reshape(1,2), self.U[k, :], self.ulim[1].reshape(1,2))) # control input constraint
            self.opti.subject_to(self.opti.bounded(self.xlim[0, 0:2].reshape(1,2), self.X[k, 0:2], self.xlim[1, 0:2].reshape(1,2))) # state constraint
            # 将self.xlim从第3个元素往后的数组，通过reshape(1,-1)重整成1行，自动确定列数大小的数组
            self.opti.subject_to(self.opti.bounded(self.xlim[0, 2:].reshape(1,-1), self.X[k, 3:], self.xlim[1, 2:].reshape(1,-1))) # state constraint
            # 添加障碍物 avoidance 约束。它首先定义了一个辅助变量 self.s[k]，用于存储第 k 步的 avoidance 成本
            for g in self.obsAvoid(self.obstacle_list, self.X[k,:]):
                self.opti.subject_to(g <= self.s[k])
            # constraint_error = self.slackObsAvoid(self.obstacle_list, self.X[k, :])
            cost += ca.mtimes([self.s[k], self.M, self.s[k]])
            
        # 计算控制量的目标值和实际值之间的差异，然后将它们与速度和角度的误差进行拼接。
        terminal_state_error = ca.horzcat(
                self.X[self.N, :2] - self.X_ref[self.N, :2],
                self.angleDiff(self.X[self.N, 2], self.X_ref[self.N, 2]),
                self.X[self.N, 3:] - self.X_ref[self.N, 3:]
            )
        # print('terminal_state_error,',terminal_state_error.shape)
        cost += ca.mtimes([terminal_state_error, self.P, terminal_state_error.T])
        self.opti.subject_to(self.X[0, :] == self.X_init)# Initial state as constraints
        self.opti.subject_to(self.opti.bounded(self.xlim[0, 0:2].reshape(1,2), self.X[self.N, 0:2], self.xlim[1, 0:2].reshape(1,2))) # state constraint
        self.opti.subject_to(self.opti.bounded(self.xlim[0, 2:].reshape(1,-1), self.X[self.N, 3:], self.xlim[1, 2:].reshape(1,-1))) # state constraint
        for g in self.obsAvoid(self.obstacle_list, self.X[self.N,:]):
                self.opti.subject_to(g <= self.s[self.N])
        # self.opti.subject_to(self.obsAvoid(self.obstacle_list, self.X[self.N,:]) <= self.s[self.N])
        # terminal_constraint_error = self.slackObsAvoid(self.obstacle_list, self.X[self.N, :])
        cost += ca.mtimes([self.s[self.N], self.M, self.s[self.N]])
        
        self.opti.minimize(cost) # 调用优化器

        
        
        # Set solver options
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}
        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_init, traj_ref, u_ref):

        # Set initial guess for the optimization problem
        if self.X_guess is None:
            self.X_guess = np.ones((self.N+1, 6)) * x_init

        if self.U_guess is None:
            self.U_guess = np.zeros((self.N, 2))
        
        self.opti.set_initial(self.X, self.X_guess)
        self.opti.set_initial(self.U, self.U_guess)
        self.opti.set_initial(self.s, np.zeros((self.N+1, 1)))

        # 将参考轨迹（traj_ref）设置到优化器中的参考状态量变量（self.X_ref）中
        self.opti.set_value(self.X_ref, traj_ref)
        # 将参考控制量（u_ref）设置到优化器中的参考控制量变量（self.U_ref）中
        self.opti.set_value(self.U_ref, u_ref)
        
        self.opti.set_value(self.X_init, x_init)

        try:
            sol = self.opti.solve()
            # s = self.opti.debug.value(self.s)
        except:
            print("here should be a debug breakpoint")
            x = self.opti.debug.value(self.X)
            for x_k in x:
                print(self.obsAvoid(self.obstacle_list, x_k))
            print("x:", self.opti.debug.value(self.X))
            print("y:", self.opti.debug.value(self.U))
            print("s:", self.opti.debug.value(self.s))
        
        ## obtain the initial guess of solutions of the next optimization problem
        self.X_guess = sol.value(self.X)
        self.U_guess = sol.value(self.U) 
        if self.U_guess.ndim == 1: 
            self.U_guess = self.U_guess.reshape(-1,2)
        return self.U_guess[0, :]
        


