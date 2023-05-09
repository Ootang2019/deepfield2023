import time

import gym
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from gym import spaces
from std_srvs.srv import Empty, EmptyRequest
from turtlesim.msg import Pose
from turtlesim.srv import (
    Kill,
    KillRequest,
    Spawn,
    SpawnRequest,
    TeleportAbsolute,
    TeleportAbsoluteRequest,
)


def lmap(v, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


class TurtleEnv(gym.Env):
    def __init__(
        self,
        index=1,  # env index
        rate=100,  # simulation refresh rate
    ):
        super().__init__()
        self.index = index

        rospy.init_node("TurtleEnv" + str(index))

        self.rate = rospy.Rate(rate)
        self._create_ros_pub_sub_srv()

        self.act_dim = 2  # [linear.x, angular.z]
        self.action_space = spaces.Box(
            low=-np.ones(self.act_dim), high=np.ones(self.act_dim), dtype=np.float32
        )
        self.act_bnd = {"linear_x": (0, 5), "ang_z": (-10, 10)}

        self.obs_dim = 7  # [dist, theta, cos(theta), sin(theta)]
        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32
        )
        self.obs_bnd = {"xy": 11.1}  # wall position [0,11.1]

        self.rew_w = np.array(
            [1, 0, 0, 0, 0, 0, 0]
        )  # [dist, delta_theta, cos(delta_theta), sin(delta_theta), x, y, theta]
        self.dist_threshold = 0.1  # done if dist < dist_threshold

        self.goal_name = "turtle2"
        self.pos = np.zeros(3)

        self.steps = 0
        self.total_ts = 0

        self.reset()
        rospy.loginfo("[ ENV Node " + str(self.index) + " ] Initialized")

    def step(self, action):
        if not rospy.is_shutdown():
        
            self.steps += 1
            self.total_ts += 1

            self.publish_action(action)
            observation = self.observe()
            reward = self.compute_reward(observation)
            done = self.is_terminal(observation)
            info = {}

            self.rate.sleep()
            return observation, reward, done, info
        else:
            rospy.logerr("rospy is shutdown")

    def reset(self):
        self.steps = 0
        try:
            self._clear_goal()
        except:
            None
        self._reset_background()
        self._reset_turtle()
        self.set_goal()
        return self.observe()

    def publish_action(self, action):
        action = self._proc_action(action)
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        self.action_publisher.publish(cmd)

    def observe(self, n_std=0.2):
        relative_dist = np.linalg.norm(self.pos[0:2] - self.goal[0:2], 2)
        relative_ang = self.compute_angle(self.goal, self.pos)
        rel = np.array(
            [relative_dist, relative_ang, np.sin(relative_ang), np.cos(relative_ang)]
        )
        return np.concatenate([rel, self.pos]) + np.random.normal(
            0, n_std, self.obs_dim
        )

    def compute_reward(self, obs):
        return np.dot(self.rew_w, -np.abs(obs))

    def set_goal(self):
        self.goal = self._random_goal()

        request = SpawnRequest()
        request.name = self.goal_name
        request.x, request.y = self.goal[0], self.goal[1]
        request.theta = self.goal[2]
        self.spawn_srv(request)

    def is_terminal(self, observation):
        done = observation[0] <= self.dist_threshold
        return done

    def _create_ros_pub_sub_srv(self):
        self.action_publisher = rospy.Publisher(
            "sim" + str(self.index) + "/turtle1/cmd_vel", Twist, queue_size=1
        )
        rospy.Subscriber(
            "sim" + str(self.index) + "/turtle1/pose", Pose, self._pose_callback
        )

        self.clear_srv = rospy.ServiceProxy("/sim" + str(self.index) + "/clear", Empty)
        self.kill_srv = rospy.ServiceProxy("/sim" + str(self.index) + "/kill", Kill)
        self.spawn_srv = rospy.ServiceProxy("/sim" + str(self.index) + "/spawn", Spawn)
        self.teleport_srv = rospy.ServiceProxy(
            "/sim" + str(self.index) + "/turtle1/teleport_absolute",
            TeleportAbsolute,
        )
        time.sleep(0.1)

    def _pose_callback(self, msg: Pose):
        self.pos = np.array([msg.x, msg.y, msg.theta])

    def _proc_action(self, action, noise_std=0.1):
        proc = action + np.random.normal(0, noise_std, action.shape)
        proc = np.clip(proc, -1, 1)
        proc[0] = lmap(proc[0], [-1, 1], self.act_bnd["linear_x"])
        proc[1] = lmap(proc[1], [-1, 1], self.act_bnd["ang_z"])
        return proc

    def _random_goal(self):
        goal = np.random.uniform(-1, 1, 3)
        goal[0:2] = self.obs_bnd["xy"]*0.5*(goal[0:2]+1)
        goal[2] = np.pi*goal[2]
        return goal

    def _clear_goal(self):
        kill_obj = KillRequest()
        kill_obj.name = self.goal_name
        self.kill_srv(kill_obj)

    def _reset_background(self):
        self.clear_srv(EmptyRequest())

    def _reset_turtle(self):
        teleport_obj = TeleportAbsoluteRequest()
        teleport_obj.x = 5.5
        teleport_obj.y = 5.5
        teleport_obj.theta = np.pi * np.random.uniform(-1, 1)
        self.teleport_srv(teleport_obj)

    @classmethod
    def compute_angle(cls, goal_pos: np.array, obs_pos: np.array) -> float:
        pos_diff = obs_pos - goal_pos
        goal_yaw = np.arctan2(pos_diff[1], pos_diff[0]) - np.pi
        ang_diff = goal_yaw - obs_pos[2]

        if ang_diff > np.pi:
            ang_diff -= 2 * np.pi
        elif ang_diff < -np.pi:
            ang_diff += 2 * np.pi

        return ang_diff

    def render(self):
        raise NotImplementedError

    def close(self):
        rospy.signal_shutdown("Training Complete") 


class TurtleEnv_Hard(TurtleEnv):
    """dynamic 2nd turtle"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def publish_action(self, action):
        action = self._proc_action(action)
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        self.action_publisher.publish(cmd) 

        cmd.linear.x *=0.5  
        cmd.angular.z *=-0.5
        self.goal_action_publisher.publish(cmd)        

    def _create_ros_pub_sub_srv(self):
        self.action_publisher = rospy.Publisher(
            "sim" + str(self.index) + "/turtle1/cmd_vel", Twist, queue_size=1
        )
        self.goal_action_publisher = rospy.Publisher(
            "sim" + str(self.index) + "/turtle2/cmd_vel", Twist, queue_size=1
        )

        rospy.Subscriber(
            "sim" + str(self.index) + "/turtle1/pose", Pose, self._pose_callback
        )

        self.clear_srv = rospy.ServiceProxy("/sim" + str(self.index) + "/clear", Empty)
        self.kill_srv = rospy.ServiceProxy("/sim" + str(self.index) + "/kill", Kill)
        self.spawn_srv = rospy.ServiceProxy("/sim" + str(self.index) + "/spawn", Spawn)
        self.teleport_srv = rospy.ServiceProxy(
            "/sim" + str(self.index) + "/turtle1/teleport_absolute",
            TeleportAbsolute,
        )
        time.sleep(0.1)



if __name__ == "__main__":
    rospy.init_node("TurtleEnv")

    env = TurtleEnv()
    env.reset()
    for _ in range(100000):
        action = env.action_space.sample()
        action = np.ones_like(action)  # [thrust, ang_vel]
        obs, reward, terminal, info = env.step(action)
