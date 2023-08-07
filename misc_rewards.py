import numpy as np

from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
import values


class EventReward(RewardFunction):
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward


class VelocityReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED * (1 - 2 * self.negative)


class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount)


class ConstantReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 1


class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)

        #print(f"DR: {defensive_reward}, OR: {offensive_reward}")

        return defensive_reward + offensive_reward

#------------------------------------------------------------
#Written by me 
class GoToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        distance = np.linalg.norm(x=pos_diff, ord=2)
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        angle = float(np.dot(player.car_data.forward(), norm_pos_diff))

        reward = 2000 if distance <= 200 else 0
        reward += 10/np.abs(angle)

        if -2 < angle <2:
            reward += 1000/distance

        # if distance <=200:
        #     print(f"Distance: {distance} Angle: {angle} Reward: {reward}")

        return reward

class DistanceTOGoalReward(RewardFunction):
    def __init__(self, distance_weight=0., diameter_scale = 0.) -> None:
        self.distance_weight = distance_weight 
        self.diameter_scale = diameter_scale
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        distance_to_orange_goal = np.linalg.norm(x=values.ORANGE_GOAL_BACK - state.ball.position, ord=2)
        distance_to_blue_goal = np.linalg.norm(x=values.BLUE_GOAL_BACK - state.ball.position, ord=2)
        # dtb: Larger value when ball is closer to the blue goal
        dtb = ((self.diameter_scale*values.BALL_RADIUS) / (distance_to_blue_goal)) ** self.distance_weight 
        # dtb: Larger value when ball is closer to the orange goal
        dto = ((self.diameter_scale*values.BALL_RADIUS) / (distance_to_orange_goal)) ** self.distance_weight

        if player.team_num == 0: #Blue team
            reward = dto - dtb
            
        else: #Orange Team
            reward = dtb - dto

        return reward
