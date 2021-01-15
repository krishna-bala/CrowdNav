import rospy
import numpy as np
import time
from tf import transformations
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from gazebo_msgs.msg import ModelState, ModelStates
import configparser
import gym
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionXY

from MPC import traj_opt
import copy
from matplotlib import pyplot as plt


class SARLPolicy:

    def __init__(self, robot_data):

        self.robot_radius = robot_data['radius']
        self.robot_pref_speed = robot_data['pref_speed']  # TODO: Make this a dynamic variable
        self.robot_mpc = robot_data['mpc']
        if self.robot_mpc:
            self.dt = 0.1
            self.times_steps = int(1.0 / self.dt)
            self.mpc = traj_opt(dt=self.dt, time_steps=self.times_steps)
            self.mpc_state = ModelState()
            self.mpc_psi = None

        self.stop_moving_flag = False
        self.state = ModelState()
        self.STATE_SET_FLAG = False
        self.goal = PoseStamped()
        self.GOAL_RECEIVED_FLAG = False
        self.GOAL_THRESH = 0.5

        # External Agent(s) state
        self.other_agents_state = [ObservableState(float("inf"), float("inf"), 0, 0, 0.3)]
        self.OBS_RECEIVED_FLAG = False

        # what we use to send commands
        self.desired_action = ActionXY(0, 0)

    def compute_action(self):

        tic = time.time()
        # Set robot
        px, py = self.state.pose.position.x, self.state.pose.position.y
        vx, vy = self.state.twist.linear.x, self.state.twist.linear.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        q = self.state.pose.orientation
        _, _, yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        radius, v_pref = self.robot_radius, self.robot_pref_speed
        robot.set(px, py, gx, gy, vx, vy, yaw, radius, v_pref)
        robot_dist_to_goal = np.linalg.norm(np.array([px, py]) - np.array([gx, gy]))

        if robot_dist_to_goal < self.GOAL_THRESH:
            self.stop_moving_flag = True
            return Twist()
        else:
            self.stop_moving_flag = False

        ### INFO ###
        # print "\n================================================================\n"
        # print "SARLPolicyNode.compute_action:"
        # print "--->self.vel.x: ", self.state.twist.linear.x
        # print "--->self.vel.y: ", self.state.twist.linear.y
        # print "--->self.vel: ", np.linalg.norm([self.state.twist.linear.x, self.state.twist.linear.y])
        ### INFO ###

        self.desired_action = robot.act(self.other_agents_state)
        twist = Twist()
        twist.linear.x = np.cos(yaw)*self.desired_action.vx + np.sin(yaw)*self.desired_action.vy
        twist.linear.y = -np.sin(yaw)*self.desired_action.vx + np.cos(yaw)*self.desired_action.vy

        ### INFO ###
        # print "\n--->SARLPolicyNode.compute_action runtime: ", time.time() - tic
        # print "\n================================================================\n"
        ### INFO ###

        return twist

    def compute_mpc(self):
        plot_flag = False
        self.mpc_state.pose.position.x = self.state.pose.position.x
        self.mpc_state.pose.position.y = self.state.pose.position.y
        self.mpc_state.twist.linear.x = self.state.twist.linear.x
        self.mpc_state.twist.linear.y = self.state.twist.linear.y

        global other_agents
        propagated_agents_state = self.mpc.generate_obs_positions(copy.deepcopy(other_agents))

        q = self.state.pose.orientation
        _, _, yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.mpc_psi = yaw

        ref_traj = [[0, 0, self.mpc_state.twist.linear.x, 0, 0, self.mpc_state.twist.linear.y,
                     self.mpc_state.pose.position.x, self.mpc_state.pose.position.y]]

        for i in range(self.times_steps):
            other_agents_copy = other_agents_from_states(propagated_agents_state[i])
            action, stop_moving_flag = self.rollout_mpc(other_agents_copy)
            if stop_moving_flag:
                break
            curr_pos, curr_vel = self.compute_mpc_state(action)
            ref_traj.append([0, 0, curr_vel[0], 0, 0, curr_vel[1], curr_pos[0], curr_pos[1]])

        if (len(ref_traj) < self.times_steps + 1):
            last_state = ref_traj[-1]
            for i in range(self.time_steps + 1 - len(ref_traj)):
                ref_traj.append(last_state)

        s_opt, u_opt = self.mpc.solv(ref_traj, get_s_opt=plot_flag)
        if (plot_flag):
            u_opt = u_opt[0]
            ref_traj = np.array(ref_traj)
            plt.plot(ref_traj[:,6],ref_traj[:,7])
            plt.plot(s_opt[:,0],s_opt[:,1])
            plt.show()

        mpc_twist = Twist()
        mpc_twist.linear.x = u_opt[0] * np.cos(yaw) + u_opt[1] * np.sin(yaw)
        mpc_twist.linear.y = -1 * u_opt[0] * np.sin(yaw) + u_opt[1] * np.cos(yaw)

        return mpc_twist

    def compute_mpc_state(self, action):

        vx = np.cos(self.mpc_psi)*action.vx + np.sin(self.mpc_psi)*action.vy
        vy = -np.sin(self.mpc_psi)*action.vx + np.cos(self.mpc_psi)*action.vy

        self.mpc_state.pose.position.x = self.mpc_state.pose.position.x + vx*self.dt
        self.mpc_state.pose.position.y = self.mpc_state.pose.position.y + vy*self.dt
        self.mpc_state.twist.linear.x = vx
        self.mpc_state.twist.linear.y = vy

        return [self.mpc_state.pose.position.x, self.mpc_state.pose.position.y], [vx, vy]

    def rollout_mpc(self, other_agents_copy):
        px = self.mpc_state.pose.position.x
        py = self.mpc_state.pose.position.y
        vx = self.mpc_state.twist.linear.x
        vy = self.mpc_state.twist.linear.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y
        q = self.mpc_state.pose.orientation
        _, _, yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        radius, v_pref = self.robot_radius, self.robot_pref_speed

        # in case current speed is larger than desired speed
        v = np.linalg.norm(np.array([vx, vy]))
        if v > v_pref:
            vx = vx * v_pref / v
            vy = vy * v_pref / v

        robot.set(px, py, gx, gy, vx, vy, yaw, radius, v_pref)

        mpc_action = robot.act(other_agents_copy)

        robot_dist_to_goal = np.linalg.norm(np.array([px, py]) - np.array([gx, gy]))

        if robot_dist_to_goal < self.GOAL_THRESH:
            stop_moving_flag = True
        else:
            stop_moving_flag = False

        return mpc_action, stop_moving_flag

    def generate_twist(self):
        if self.robot_mpc:
            self.compute_mpc()
        else:
            self.compute_action()

    def update_state(self, robot_state):
        self.state = robot_state
        self.STATE_SET_FLAG = True

    def update_dynamic_goal(self, msg):
        self.GOAL_RECEIVED_FLAG = True
        new_goal = PoseStamped()
        new_goal.pose.position.x = msg.pose.position.x
        new_goal.pose.position.y = msg.pose.position.y
        self.goal = new_goal

    def set_other_agents(self, humans):
        self.OBS_RECEIVED_FLAG = True
        self.other_agents_state = humans


def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def cb_state(msg):
    global state
    global STATE_SET
    state.pose.position = msg.pose.position
    state.twist.linear = msg.twist.linear
    STATE_SET = True


def cb_other_agents(msg):
    # Create list of HUMANS
    global other_agents
    global OTHER_AGENTS_SET
    other_agents = []
    num_agents = len(msg.name)
    for i in range(num_agents):
        radius = 0.3  # Spheres in gazebo
        x = msg.pose[i].position.x
        y = msg.pose[i].position.y
        vx = msg.twist[i].linear.x
        vy = msg.twist[i].linear.y
        other_agents.append(ObservableState(x, y, vx, vy, radius))
    OTHER_AGENTS_SET = True


def other_agents_from_states(states):
    other_agents_copy = []
    num_agents = len(states)
    for i in range(num_agents):
        radius = 0.3  # Spheres in gazebo
        x = states[i][0]
        y = states[i][1]
        vx = states[i][2]
        vy = states[i][3]
        other_agents_copy.append(ObservableState(x, y, vx, vy, radius))
    return other_agents_copy


def cb_dynamic_goal(msg):
    global goal
    global GOAL_SET
    goal = msg
    GOAL_SET = True


def cb_real_other_agent(msg):
    global other_agents
    global OTHER_AGENTS_SET
    global other_agent_prev_time_stamp
    x = msg.pose.position.x
    y = msg.pose.position.y
    vx = 0
    vy = 0
    curr_time_stamp = msg.header.stamp.secs + (msg.header.stamp.nsecs * 1e-9)
    if(OTHER_AGENTS_SET):
        vx = (other_agents[0].px - x)/(curr_time_stamp - other_agent_prev_time_stamp)
        vy = (other_agents[0].py - y)/(curr_time_stamp - other_agent_prev_time_stamp)
    other_agent_prev_time_stamp = curr_time_stamp
    other_agents = []
    other_agents.append(ObservableState(x,y,vx,vy,0.3))
    OTHER_AGENTS_SET = True


def cb_real_pose(msg):
    global state
    global state_prev_time_stamp
    global STATE_SET
    curr_time_stamp = msg.header.stamp.secs + (msg.header.stamp.nsecs * 1e-9)
    if(STATE_SET):
        numx = msg.pose.position.x - state.pose.position.x
        den = curr_time_stamp - state_prev_time_stamp
        numy = msg.pose.position.y - state.pose.position.y
        state.twist.linear.x = numx/den
        state.twist.linear.y = numy/den
    state.pose = msg.pose
    state_prev_time_stamp = curr_time_stamp
    STATE_SET = True


def initialize_robot():
    model_dir = "crowd_nav/data/output-lab-base-cases/"
    phase = "test"
    model_weights = model_dir + "rl_model.pth"
    policy_config_file = model_dir + "policy.config"
    env_config_file = model_dir + "env.config"
    cuda = raw_input("Set device as Cuda? (y/n)")
    if torch.cuda.is_available() and cuda == 'y':
        device = torch.device("cuda:0")
        print "================================"
        print "=== Device: ", device, "==="
        print "================================"
    else:
        device = torch.device("cpu")
        print "===================="
        print "=== Device: ", device, "==="
        print "===================="

    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    policy = policy_factory["sarl"]()
    policy.configure(policy_config)

    if policy.trainable:
        print "SETTING MODEL WEIGHTS"
        policy.get_model().load_state_dict(torch.load(model_weights))

    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)

    env.set_robot(robot)

    policy.set_phase(phase)
    policy.set_device(device)
    policy.set_env(env)
    # TODO: NEED TO SET POLICY TIME_STEP

    return robot


if __name__ == '__main__':
    print('About to run SARL...')

    # SARL specific intializations
    robot = initialize_robot()

    try:
        state = ModelState()
        state_prev_time_stamp = None
        other_agent_prev_time_stamp = None
        STATE_SET = False
        goal = PoseStamped()
        GOAL_SET = False
        other_agents = []
        OTHER_AGENTS_SET = False

        robot_data = {'goal': None, 'radius': 0.3, 'pref_speed': 0.8, 'name': 'balabot', 'mpc': True}

        scenario = input("Running in real or gazebo?\n (1 for real, 2 for gazebo): ")
        mpc = input("Layer MPC on top? (y/n): ")

        if mpc[0].lower() == 'y':
            robot_data['mpc'] = True
        else:
            robot_data['mpc'] = False

        sarl_policy_node = SARLPolicy(robot_data)
        rospy.init_node(robot_data['name'], anonymous=False)
        rate = rospy.Rate(10)
        node_name = rospy.get_name()
        control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        if scenario == 2:
            sub_other_agents = rospy.Subscriber('/sphere_states', ModelStates, cb_other_agents)
            sub_pose = rospy.Subscriber('/sim_pathbot_state', ModelState, cb_state)
            sub_goal = rospy.Subscriber('/sim_pathbot_goal', PoseStamped, cb_dynamic_goal)
        else:
            sub_goal = rospy.Subscriber('/vrpn_client_node/agent1/pose', PoseStamped, cb_dynamic_goal)
            sub_other_agent = rospy.Subscriber('/vrpn_client_node/agent2/pose', PoseStamped, cb_real_other_agent)
            sub_pose = rospy.Subscriber('/vrpn_client_node/pathbot/pose', PoseStamped, cb_real_pose)

        while not rospy.is_shutdown():
            if STATE_SET and GOAL_SET and OTHER_AGENTS_SET:
                sarl_policy_node.update_state(state)
                sarl_policy_node.update_dynamic_goal(goal)
                sarl_policy_node.set_other_agents(other_agents)
                control_pub.publish(sarl_policy_node.generate_twist())
            rate.sleep()
    except rospy.ROSInterruptException, e:
        raise e
