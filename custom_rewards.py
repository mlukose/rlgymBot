from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
import values
import numpy as np

class CloseToBallReward(RewardFunction):
    def __init__(self, distance_weight=0., diameter_scale=0., have_negative_rewards=False):
        self.distance_weight = distance_weight # Exponent to scale reward by 
        self.diameter_scale = diameter_scale # Value to multiple ball diameter by to determine what distance should have a reward value greater than 1
        # (i.e if diameter_scale = 2, the car must be at or within 2 ball diameters of the ball for the reward to be greater than 1)
        self.have_negative_rewards = have_negative_rewards

    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        # Increases reward exponentially depending on how close the agent is to the ball. Exponent is passed in.
        # The larger the exponent the smaller the reward for large distances from the ball and the higher reward for low distances.
        pos_diff = state.ball.position - player.car_data.position
        distance = np.linalg.norm(x=pos_diff, ord=2)
        if distance == 0: # Im pretty sure distance cant be 0 b/c the min distance should be the ball radius (92.75) but this is just in case it can to avoid dividing by 0
            distance=1
        reward = ((self.diameter_scale*values.BALL_RADIUS) / (distance)) ** self.distance_weight
        #print(f"reward: {reward} \t Distance: {distance}")
        if self.have_negative_rewards: # Makes reward negative 5 if the car is too far from the ball
            if distance > (self.diameter_scale) * 2:
                reward = -5 
            elif distance > (self.diameter_scale) * 1.75:
                reward = -2

        return reward
        
class GoalScoredReward(RewardFunction):
    def __init__(self) -> None:
        self.last_registered_scores = [0,0] # [blue, orange]
    def get_current_scores(self, state: GameState):
        blue, orange = state.blue_score, state.orange_score
        return np.array([blue, orange])
        
    def reset(self, initial_state: GameState):
        pass
        


    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        old_blue_score = self.last_registered_scores[0]
        old_orange_score = self.last_registered_scores[1]
        BLUE_SCORED = False
        ORANGE_SCORED = False
        if state.blue_score - old_blue_score > 0:
            BLUE_SCORED = True
        if state.orange_score - old_orange_score > 0: #Opponent scored
            ORANGE_SCORED = True

        if player.team_num == 0: #Blue team 
            if BLUE_SCORED:
                reward = 200
            elif ORANGE_SCORED:
                reward = -200
            
        if player.team_num == 1: #Orange Team
            if ORANGE_SCORED:
                reward = 200
            elif BLUE_SCORED:
                reward = -200
            
            self.last_registered_scores[0] = state.blue_score
            self.last_registered_scores[1] = state.orange_score
            # The update must be in this if statement b/c the score update would
            # Occur before the orange player calls get_reward causing no reward
            # to be given for the orange player.
            
        return reward


class TouchBallNose(RewardFunction): # Rewards when ball is touched by nose of car 
    def __init__(self, min_angle = 0.) -> None:
        self.min_angle = min_angle
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        angle = float(np.dot(player.car_data.forward(), norm_pos_diff))
        if player.ball_touched:
            if self.min_angle <= angle <= 1:
                return 50
        return 0
    
    
