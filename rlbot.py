import sys
from os.path import exists
import rlgym
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv
import numpy as np
from numpy import random
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import *
from rlgym.utils import math
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, NoTouchTimeoutCondition
from rlgym.utils.action_parsers import ActionParser
import gym.spaces
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import *
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward
from misc_rewards import *
from custom_rewards import *
from conditional_rewards import *
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z





#Observation Builder class
class CustomObsBuilder(ObsBuilder):
  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
    obs = []
    obs += state.ball.serialize()
    
    for player in state.players:
      obs += player.car_data.serialize()
    
    return np.asarray(obs, dtype=np.float32)

#Reward function class
class rewardFunc(RewardFunction):
  def __init__(self) -> None:
     self.touch_ball = TouchBallBigReward()
     #self.go_to_ball = GoToBallReward()
     self.dist_to_goal = DistanceTOGoalReward(distance_weight=1.3, diameter_scale=80)
     self.closer_to_ball = CloseToBallReward(distance_weight=1.6, diameter_scale=10, have_negative_rewards=False)
     self.player_to_ball_vel = VelocityPlayerToBallReward()
     self.goal_scored = GoalScoredReward()
     self.align_to_goal = AlignBallGoal(defense=20, offense=20)
     #self.event_ = EventReward(goal=200, concede=-200)
     self.face_ball = FaceBallReward()
     self.ball_to_goal_vel = VelocityBallToGoalReward()
     self.nose_touch = TouchBallNose(min_angle=0.85)
     self.car_velocity = VelocityReward()

     return super().__init__()
  
  def reset(self, initial_state: GameState):
     pass


  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

    dist_to_goal_reward = self.dist_to_goal.get_reward(player, state, previous_action)
    align_to_goal_reward = self.align_to_goal.get_reward(player, state, previous_action)
    player_to_ball_vel_reward = self.player_to_ball_vel.get_reward(player, state, previous_action)
    touch_ball_reward = self.touch_ball.get_reward(player, state, previous_action)
    closer_to_ball_reward = self.closer_to_ball.get_reward(player, state, previous_action)
    goal_scored_reward = self.goal_scored.get_reward(player, state, previous_action)
    face_ball_reward = self.face_ball.get_reward(player, state, previous_action)
    ball_to_goal_velocity_reward = self.ball_to_goal_vel.get_reward(player, state, previous_action)
    nose_touch_reward = self.nose_touch.get_reward(player, state, previous_action)
    car_velocity_reward = self.car_velocity.get_reward(player, state, previous_action)

    

    
    group_1 = (player_to_ball_vel_reward*5) + (face_ball_reward*2) + nose_touch_reward + (closer_to_ball_reward/6)
    group_2 = goal_scored_reward + dist_to_goal_reward

    #print(f"Dist: {dist_to_goal_reward}, Group2: {group_2}, group 1: {group_1}")
    return group_1 + (2*group_2)
    

    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    return 0


# # Terminal Condition class
# class CustomTerminalCondition(TerminalCondition):
#   def reset(self, initial_state: GameState):
#     pass

#     #Terminates when a goal is scored for either team
#   def is_terminal(self, current_state: GameState) -> bool:
#     return current_state.blue_score or current_state.orange_score

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 75

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

condition1 = NoTouchTimeoutCondition(max_steps=max_steps)
condition2 = GoalScoredCondition() # Terminates when goal is scored by either team


class DiscreteAction(ActionParser):
    """
    Simple discrete action space. All the analog actions have 3 bins by default: -1, 0 and 1.
    """

    def __init__(self, n_bins=3):
        super().__init__()
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([self._n_bins] * 5 + [2] * 3)

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 8))

        # map all ternary actions from {0, 1, 2} to {-1, 0, 1}.
        actions[..., :5] = actions[..., :5] / (self._n_bins // 2) - 1

        return actions

#Custom State Setter
class CustomStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
       for car in state_wrapper.cars:
          # Placing cars randomly on the field
          car.position = [random.uniform(-3072.0, 3072.0), random.uniform(-4096.0, 4096.0), 17.0]
          # Randomly setting the yaw
          car.rotation = [0.0, random.uniform(-np.pi, np.pi), 0.0]
          # Setting boost
          car.boost == 50

       # Setting ball position randomly   
       state_wrapper.ball.position = [random.uniform(-1000, 1000), random.uniform(-1000, 1000), 93.0]
       # Setting Ball linear velocity randomly
       state_wrapper.ball.linear_velocity = [random.uniform(-100,100), random.uniform(-100,100), 0]
       state_wrapper.ball.angular_velocity = [random.uniform(-100,100), random.uniform(-100,100), 0]
       

       
       
#Make the default rlgym environment
gym_env = rlgym.make(game_speed=150, use_injector=True, self_play=True, state_setter=CustomStateSetter(), action_parser=DiscreteAction(), reward_fn=rewardFunc(), terminal_conditions=[condition1, condition2])
env = SB3SingleInstanceEnv(gym_env)

#Initialize PPO from stable_baselines3

# Training and Predicting loop
filepath = "C:/Users/micst/Desktop/Genetic Learning Personal Project/model_13.pt"
if sys.argv[1] == "training":
  if exists(path=filepath):
      learning_rate = 0.00001
      custom_object = {'learning_rate':learning_rate}
      model = PPO.load(path=filepath, env=env, device="cpu", custom_objects=custom_object)
  else:
      model = PPO(policy="MlpPolicy", env=env, verbose=1, device="cpu", learning_rate=0.0001, batch_size=4096)
  while True:
    model.learn(total_timesteps=int(1e5), progress_bar=True) 
    model.save(path=filepath)

if sys.argv[1] == "predicting":
   model = PPO.load(path=filepath, env=env, device="cpu")
   for _ in range(1000):
      observation = env.reset()
      steps = 0
      done = False
      while not done:
          actions = model.predict(observation)[0]
          observation, reward, done, gameinfo = env.step(actions)
          done = done.any()
          steps = steps + 1
          