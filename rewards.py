"""Custom reward logic classes for goalie-focused RLGym v2 training."""

from typing import Any, Dict, List  # Type hints for readability (says what types are expected)
import numpy as np  # for vector math (positions, velocities, directions)

# Generic reward interface used by RLGym (AgentID = identifies each player (bot), RewardFunction = base class we must inherit from to define custom reward logic)
from rlgym.api import AgentID, RewardFunction  
from rlgym.rocket_league import common_values  # Field and physics constants (goal location, max ball speed, field size,etc.)
from rlgym.rocket_league.api import GameState  # Runtime state object for one environment step (everything happening in the game at a single timestep)


# converts a vector into a unit vector (length = 1) safely
# takes in: np array representing a vector (e.g. direction from goal to ball), and safety number to prevent division by zero
# returns: np array representing the same direction with length 1
def _safe_unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Prevent divide-by-zero when two objects are at the same position
    norm = np.linalg.norm(vector)  # gets magnitude (length) of the vector
    # If vector is too small, return zero vector instead of dividing 
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


# gets Y-coordinate of the goal this car should defend
# takes in: boolean indicating if the car is on the orange team (True) or blue team (False)
# returns: Y position of the goal in field coordinates (positive for orange, negative for blue)
def _own_goal_y(is_orange: bool) -> float:
    # Blue defends negative Y, orange defends positive Y
    return common_values.BACK_WALL_Y if is_orange else -common_values.BACK_WALL_Y


# GoalieSparseReward: gives +1 reward for touching the ball on your defending half, 0 otherwise
class GoalieSparseReward(RewardFunction[AgentID, GameState, float]):
    """Sparse reward: only reward a defensive touch on agents own half to encourage staying in the goal area."""

    # Defines what counts as “defending half” based on ball Y position (1500 units from center line in our case)
    def __init__(self, defending_half_threshold: float = 1500.0):
        self.defending_half_threshold = defending_half_threshold

    # Called at the start of each episode to clear out old data. Doesn't return a value.
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:

        # Ignore these variables since there's no memory to reset for this reward function as it doesn't keep track of anything across steps (because it's purely based on current step's ball touches and position).
        # del just avoids unused variable warnings
        del agents, initial_state, shared_info

    # Called at every simulation step to calculate rewards based on current game state. Returns a dict mapping each agent to its reward for this step.
    # +1.0 when a car touches the ball on its defending side. 0.0 otherwise.
    def get_rewards(
        self,
        agents: List[AgentID],               # list of players currently in the game
        state: GameState,                    # current state of the game (positions, velocities, ball touches, etc.)
        is_terminated: Dict[AgentID, bool],  # indicates if the episode has ended for each agent (e.g. goal scored, timeout)
        is_truncated: Dict[AgentID, bool],   # indicates if the episode has been truncated (e.g. timeout)
        shared_info: Dict[str, Any],         # any additional info shared across agents (not used in this reward function)
    ) -> Dict[AgentID, float]:

        del is_terminated, is_truncated, shared_info  # these variables are not used in this reward function, so we ignore them to avoid warnings
        rewards: Dict[AgentID, float] = {}  # initialize empty dict to store rewards for each agent

        # We use ball Y to decide whether the touch happened on the defending side
        ball_y = float(state.ball.position[1])
    
        for agent in agents:
            car = state.cars[agent]  # get the car object for this agent to check its team and ball touches

            # No touch this step means no sparse reward this step
            if not car.ball_touches:
                rewards[agent] = 0.0
                continue

            # For orange team, defending half is when ball Y > threshold. For blue team, defending half is when ball Y < -threshold.
            is_defending_half = (
                ball_y > self.defending_half_threshold if car.is_orange
                else ball_y < -self.defending_half_threshold
            )
            rewards[agent] = 1.0 if is_defending_half else 0.0  # reward 1.0 for a touch on defending half, 0.0 otherwise

        return rewards

# PhysicsDenseReward: gives a dense reward based on how much the ball's velocity points away from the agent's own goal (encourages clearing the ball out of the goal area)
class PhysicsDenseReward(RewardFunction[AgentID, GameState, float]):
    """Dense reward: encourage ball velocity away from the defending goal."""

    # called at the start of each episode. This reward also does not keep internal state so we just ignore the inputs.
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:

        # No memory to reset for this reward function
        del agents, initial_state, shared_info

    # called at every simulation step to calculate rewards based on current game state. Returns a dict mapping each agent to its reward for this step.
    # Positive reward when ball velocity points away from own goal, negative when it points toward own goal. Normalized by max ball speed to keep values in a reasonable range.
    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:

        del is_terminated, is_truncated, shared_info
        rewards: Dict[AgentID, float] = {}  # empty dict for storing rewards for each agent this step

        # Get ball position and velocity as numpy arrays for vector math. 
        ball_pos = np.asarray(state.ball.position, dtype=np.float32)
        ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)

        for agent in agents:
            car = state.cars[agent]
            goal_pos = np.array([0.0, _own_goal_y(car.is_orange), 0.0], dtype=np.float32)  # own goal position based on team
            goal_to_ball_dir = _safe_unit(ball_pos - goal_pos)  # gets unit vector (direction) pointing from goal to ball (away from goal)

            # Positive reward when ball velocity points away from own goal, negative when it points toward own goal (calculated by dot product). 
            vel_away_from_goal = float(np.dot(ball_vel, goal_to_ball_dir))
            rewards[agent] = vel_away_from_goal / common_values.BALL_MAX_SPEED  # Normalized by max ball speed to keep values in a reasonable range [-1.0 to 1.0]

        return rewards

# ContextAwareReward: combines the dense physics reward with additional context-based bonuses and penalties to encourage good goalie behavior (facing the ball, managing boost, and staying near the goal)
class ContextAwareReward(RewardFunction[AgentID, GameState, float]):
    """Hybrid reward: dense physics signal plus positioning, orientation, and boost context."""

    def __init__(
        self,
        goalie_radius: float = 2200.0,  # how far you should stay near goal to avoid penalty (2200 is about the distance from goal to the midfield line, i think)
        facing_weight: float = 0.05,    # small reward for facing the ball (encourages good positioning and orientation to react to threats)
        boost_weight: float = 0.005,    # very small reward for having enough boost to react (encourages boost management, important for a goalie to be able to make saves and clear the ball)
        position_penalty_weight: float = 0.05, # # penalty for drifting too far from the goal (encourages staying in the goal area and not overcommitting forward)
        no_touch_penalty_weight: float = 0.03,  # explicit penalty for prolonged no-touch behavior
        no_touch_penalty_steps: int = 75,       # ~5s at 15 decisions/sec (action_repeat=8)
        center_ball_y_threshold: float = 1400.0,  # treat ball as "near center" when abs(y) is below this
        challenge_distance: float = 2200.0,       # distance above which agent is considered not challenging
        center_no_challenge_weight: float = 0.06, # stronger penalty when center ball is left unchallenged
        touch_bonus: float = 0.30,                # event-based positive reward on any ball contact (not farmable while idle)
    ):

        self.goalie_radius = goalie_radius
        self.facing_weight = facing_weight
        self.boost_weight = boost_weight
        self.position_penalty_weight = position_penalty_weight
        self.no_touch_penalty_weight = no_touch_penalty_weight
        self.no_touch_penalty_steps = max(1, int(no_touch_penalty_steps))
        self.center_ball_y_threshold = center_ball_y_threshold
        self.challenge_distance = challenge_distance
        self.center_no_challenge_weight = center_no_challenge_weight
        self.touch_bonus = touch_bonus

        # We reuse the dense reward so this strategy keeps a strong "clear the ball" signal while adding context-aware bonuses and penalties on top
        self._dense = PhysicsDenseReward()
        self._steps_since_touch: Dict[AgentID, int] = {}

    # called at the start of each episode to reset any internal state. This reward doesn't keep track of anything across steps, but we still need to call reset on the internal dense reward component to ensure it starts fresh each episode.
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:

        # Reset internal dense reward component (PhysicsDenseReward) to ensure it starts fresh each episode (doesn't keep state but good practice if we later modify to have state)
        self._dense.reset(agents, initial_state, shared_info)  
        self._steps_since_touch = {agent: 0 for agent in agents}

    # called at every simulation step to calculate rewards based on current game state. Returns a dict mapping each agent to its reward for this step.
    # Combines the dense physics reward with additional context-based bonuses for facing the ball and having boost, and a penalty for drifting too far from the goal.
    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:

        # Reuse dense physics signal as the base component 
        dense_rewards = self._dense.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        rewards: Dict[AgentID, float] = {}

        ball_pos = np.asarray(state.ball.position, dtype=np.float32) 

        for agent in agents:
            car = state.cars[agent]
            car_pos = np.asarray(car.physics.position, dtype=np.float32)
            car_forward = np.asarray(car.physics.forward, dtype=np.float32)  # unit vector pointing in the direction the car is facing
            to_ball_dir = _safe_unit(ball_pos - car_pos)  # unit vector pointing from car to ball (direction of the threat)

            # Bonus for facing the threat (ball), otherwise zero
            facing_bonus = self.facing_weight * max(0.0, float(np.dot(car_forward, to_ball_dir)))  

            # Small bonus for keeping enough boost to react
            has_boost_bonus = self.boost_weight if car.boost_amount > 20.0 else 0.0

            # Penalize over-committing far away from your own net
            goal_pos = np.array([0.0, _own_goal_y(car.is_orange), 0.0], dtype=np.float32)
            dist_to_goal = float(np.linalg.norm(car_pos - goal_pos))  # distance from the car to its own goal
            dist_to_ball = float(np.linalg.norm(car_pos - ball_pos))
            outside_goalie_box = max(0.0, (dist_to_goal - self.goalie_radius) / self.goalie_radius)  # 0 when within goalie_radius of the goal, scales up to 1 as you get twice as far from the goal as the goalie_radius
            position_penalty = self.position_penalty_weight * outside_goalie_box  # penalty scales up as you get farther from the goal, encourages staying in the goal area without a hard cutoff

            # Track no-touch duration to discourage stationary "camping" behavior
            had_touch = bool(car.ball_touches)
            if had_touch:
                self._steps_since_touch[agent] = 0
            else:
                self._steps_since_touch[agent] = self._steps_since_touch.get(agent, 0) + 1

            no_touch_ratio = min(1.0, self._steps_since_touch[agent] / float(self.no_touch_penalty_steps))
            no_touch_penalty = self.no_touch_penalty_weight * no_touch_ratio

            # Extra penalty if the ball sits near midfield while the agent stays far and does not challenge
            center_ratio = max(0.0, 1.0 - (abs(float(ball_pos[1])) / self.center_ball_y_threshold))
            far_from_challenge_ratio = max(0.0, (dist_to_ball - self.challenge_distance) / self.challenge_distance)
            center_no_challenge_penalty = self.center_no_challenge_weight * center_ratio * far_from_challenge_ratio * no_touch_ratio

            # Event-based positive reward for actually contacting the ball (un-hackable while idle)
            touch_reward = self.touch_bonus if had_touch else 0.0

            # final context-aware score = dense clear signal + context bonuses + touch reward - penalties
            rewards[agent] = (
                dense_rewards[agent]
                + facing_bonus
                + has_boost_bonus
                + touch_reward
                - position_penalty
                - no_touch_penalty
                - center_no_challenge_penalty
            )

        return rewards