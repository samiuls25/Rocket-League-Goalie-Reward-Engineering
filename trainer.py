"""PPO trainer for sparse, dense, and context-aware goalie reward experiments."""

import argparse  # built-in module for command-line arguments like --strategy and --n-proc
from functools import partial # allows us to create a version of build_env with the strategy argument pre-filled, which is needed for rlgym_ppo workers
from pathlib import Path  # for cleaner path handling when creating output folders for checkpoints and metrics
from typing import Callable, Dict, List, Optional  # Type hints for readability (says what types are expected), callable means var is a function
import numpy as np  # for handling numeric arrays, used in observation normalization and custom metrics extraction

from rlgym.api import RLGym  # main environment class that defines the Rocket League simulation, including state mutators, reward functions, termination conditions, and physics engine
from rlgym.rocket_league import common_values  # shared Rocket League constants (field size, ball max speed, etc.)
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction  # LookupTableAction maps policy outputs into real in-game controls, RepeatAction holds each action for multiple simulation ticks to speed up training
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, TimeoutCondition  # episode ending rules
from rlgym.rocket_league.obs_builders import DefaultObs  # builds normalized observations of game state (what the agent can "see")
from rlgym.rocket_league.rlviser import RLViserRenderer  # RLViser renderer used when --render is enabled
from rlgym.rocket_league.sim import RocketSimEngine  # physics engine backend used for fast simulation
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence  # controls how each new episode starts

from rlgym_ppo import Learner  # the main PPO training loop implementation that handles multiple workers, policy updates, checkpointing, and logging
from rlgym_ppo.util import MetricsLogger, RLGymV2GymWrapper  # wraps RLGym v2 envs to look like gym.Env, which is what rlgym_ppo expects (allows learner to interact with our game environment)
from rewards import GoalieSparseReward, PhysicsDenseReward, ContextAwareReward  # our three reward strategies for the comparison study


# Custom metrics logger to extract and log goalie-specific metrics to W&B alongside PPO's built-in metrics like reward and episode length. 
# This allows us to track things like ball speed, position, boost usage, etc. over the course of training.
class GoalieWandbMetricsLogger(MetricsLogger):
    """Custom goalie metrics logged to W&B alongside built-in PPO metrics."""

    # called by PPO learner at each step with current game state. Extract relevant goalie metrics and return them as list of numpy arrays (one per environment instance)
    def _collect_metrics(self, game_state) -> List[np.ndarray]:
        ball_pos = np.asarray(game_state.ball.position, dtype=np.float32)
        ball_vel = np.asarray(game_state.ball.linear_velocity, dtype=np.float32)

        ball_speed_norm = float(np.linalg.norm(ball_vel) / common_values.BALL_MAX_SPEED)
        ball_y_norm = float(ball_pos[1] / common_values.BACK_WALL_Y)  # tracks how close ball is to either back wall

        # Extract car-specific metrics for the tracked side only (blue team by default) so opponent behavior does not dilute goalie analysis.
        # In this 1v1 setup, blue = not orange.
        tracked_cars = [car for car in game_state.cars.values() if not car.is_orange]

        # Fallback to all cars if the tracked side is unexpectedly empty, so logging never crashes.
        cars = tracked_cars if tracked_cars else list(game_state.cars.values())
        if not cars:
            metrics = np.zeros(6, dtype=np.float32)
        else:
            boosts = np.asarray([car.boost_amount for car in cars], dtype=np.float32)  
            touches = np.asarray([float(car.ball_touches) for car in cars], dtype=np.float32)  

            own_goal_dists = [] # calculate distance from each car to its own goal
            ball_dists = []     # calculate distance from each car to the ball
            for car in cars:
                car_pos = np.asarray(car.physics.position, dtype=np.float32)
                own_goal_y = common_values.BACK_WALL_Y if car.is_orange else -common_values.BACK_WALL_Y  # determine which back wall is the car's own goal based on team
                own_goal_pos = np.array([0.0, own_goal_y, 0.0], dtype=np.float32)  # own goal position is centered on back wall at ground level
                own_goal_dists.append(np.linalg.norm(car_pos - own_goal_pos))  # get own goal distance based on positions
                ball_dists.append(np.linalg.norm(car_pos - ball_pos))  # get ball distance based on positions

            # take mean of each metric across tracked cars
            metrics = np.array(
                [
                    ball_speed_norm,
                    ball_y_norm,
                    float(np.mean(boosts)),
                    float(np.mean(touches)),
                    float(np.mean(own_goal_dists)),
                    float(np.mean(ball_dists)),
                ],
                dtype=np.float32,
            )

        return [metrics]

    # Logs the average of each metric across all workers to W&B for visualization, runs after finishing an iteration
    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        if wandb_run is None:
            return

        rows = []
        
        # Each report is a list of metric arrays (one per environment instance). 
        # We take the first one since we only have one environment instance per worker, and convert it to a numpy array for easier handling.
        for report in collected_metrics:
            if len(report) == 0:
                continue
            rows.append(np.asarray(report[0], dtype=np.float32).reshape(-1)) # reshape to ensure it's a 1D array of metrics for this environment instance

        # no metrics collected, return early
        if not rows:
            return

        # Stack all the metric arrays vertically to create a 2D array where each row corresponds to one environment instance's metrics. 
        # This allows us to easily compute the mean of each metric across all instances.
        metrics_2d = np.vstack(rows)
        wandb_run.log(
            {
                "Goalie/cumulative_timesteps": float(cumulative_timesteps),
                "Goalie/ball_speed_norm_mean": float(np.mean(metrics_2d[:, 0])),
                "Goalie/ball_y_norm_mean": float(np.mean(metrics_2d[:, 1])),
                "Goalie/mean_boost_mean": float(np.mean(metrics_2d[:, 2])),
                "Goalie/touch_rate_mean": float(np.mean(metrics_2d[:, 3])),
                "Goalie/dist_to_own_goal_mean": float(np.mean(metrics_2d[:, 4])),
                "Goalie/dist_to_ball_mean": float(np.mean(metrics_2d[:, 5])),
                "Goalie/metric_samples": float(metrics_2d.shape[0]),
            }
        )


# maps each strategy name to the reward class that creates the corresponding reward object
# this allows us to easily select which reward logic to use based on the --strategy command-line argument when we build the environment for each worker
REWARD_FACTORIES: Dict[str, Callable[[], object]] = {
    "sparse": GoalieSparseReward,
    "dense": PhysicsDenseReward,
    "context": ContextAwareReward,
}

# creates one rocket league training environment instance. Returns Gym-like wrapped environment ready for rlgym_ppo workers.
# strategy: which reward logic, action_repeat: how many simulation ticks each chosen action is held for, game_timeout_seconds: episode timeout length
def build_env(
    strategy: str,
    action_repeat: int = 8,
    game_timeout_seconds: int = 15,
    enable_render: bool = False,
):
    # Note: The PPO learner will call this function multiple times to create separate environments for each worker process
    if strategy not in REWARD_FACTORIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(REWARD_FACTORIES)}")

    # action repeat = 8 means one chosen action is held for 8 simulation ticks. speeds training, is standard.
    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)

    # pick reward logic based on --strategy flag
    reward_fn = REWARD_FACTORIES[strategy]()

    # Normalize observations [-1 to 1] so training is numerically stable/easy to learn from
    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,  # X pos / distance to side wall
                1 / common_values.BACK_NET_Y,   # Y pos / distance to back net
                1 / common_values.CEILING_Z,    # Z pos / height of ceiling
            ]
        ),
        ang_coef=1 / np.pi,  # divide angles by pi to normalize
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED, # divide car speed by max car speed 
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,  # divide car angular velocity by max 
        boost_coef=1 / 100.0,  # divide boost amount by 100.0 (max boost)
    )

    renderer = RLViserRenderer(tick_rate=120 / action_repeat) if enable_render else None

    # Environment definition:
    rlgym_env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),  # sets up 1v1 match
            KickoffMutator(),  # resets ball and cars to kickoff positions each episode
        ),
        obs_builder=obs_builder,  # passes normalized observations to the agent
        action_parser=action_parser,  # applies action repeat and maps discrete actions to real controls
        reward_fn=reward_fn,  # picks reward logic based on chosen strategy 
        termination_cond=GoalCondition(),  # ends episode when a goal is scored
        truncation_cond=AnyCondition(TimeoutCondition(timeout_seconds=game_timeout_seconds)),  # also end episode if it reaches the timeout limit
        transition_engine=RocketSimEngine(),  # use RocketSim for faster simulation
        renderer=renderer,
    )

    # Wrap to gym-like interface expected by rlgym_ppo
    return RLGymV2GymWrapper(rlgym_env)


# Command-line argument parsing to easily configure different training runs without changing code
def parse_args() -> argparse.Namespace:
    # creates the parser, description shows up when you run `python trainer.py --help`
    parser = argparse.ArgumentParser(description="Train a Rocket League goalie agent with RLGym v2.")

    parser.add_argument("--strategy", choices=sorted(REWARD_FACTORIES.keys()), default="sparse")  # to specify a strat pre-defined
    parser.add_argument("--run-name", default=None)  # optional custom name for folder results
    parser.add_argument(
        "--checkpoint-load-folder",
        default=None,
        help="Checkpoint folder to resume from, or run folder containing numeric checkpoint subfolders.",
    )

    # Throughput and hardware controls:
    parser.add_argument("--n-proc", type=int, default=8)  # how many parallel worker processes to run
    parser.add_argument("--device", default="cuda")  # where neural network training runs (cuda or cpu)
    parser.add_argument("--render", action="store_true", help="Enable RLViser rendering during rollout collection.")
    parser.add_argument("--render-delay", type=float, default=0.0, help="Seconds to wait between rendered frames.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging for this run.")

    # Output controls:
    parser.add_argument("--save-root", default="agents")  # stores model checkpoints to subfolder named after the run (e.g. agents/sparse_v1/)

    # PPO controls:
    parser.add_argument("--timestep-limit", type=int, default=40_000_000)  # total timesteps to train before stopping (40M default)
    parser.add_argument("--save-every-ts", type=int, default=300_000)  # save checkpoint every 300k timesteps
    parser.add_argument("--ppo-batch-size", type=int, default=50_000)  # number of timesteps to use for each PPO update (improve stability)
    parser.add_argument("--ppo-epochs", type=int, default=10)  # how many times to reuse each batch of experience when updating policy (improve learning)

    return parser.parse_args() # returns an object where we can access the arguments as attributes, e.g. args.strategy, etc.


def resolve_checkpoint_path(checkpoint_load_folder: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint path from either a timestep folder or a run folder."""
    if checkpoint_load_folder is None:
        return None

    checkpoint_path = Path(checkpoint_load_folder)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    # If the user passed a direct checkpoint folder, use it as-is.
    if (checkpoint_path / "BOOK_KEEPING_VARS.json").exists():
        return str(checkpoint_path)

    # Otherwise, pick the latest numeric subfolder that contains bookkeeping.
    valid_subfolders = []
    for child in checkpoint_path.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "BOOK_KEEPING_VARS.json").exists():
            valid_subfolders.append((int(child.name), child))

    if not valid_subfolders:
        raise ValueError(
            "Could not find a valid checkpoint under "
            f"{checkpoint_path}. Expected BOOK_KEEPING_VARS.json in the folder or in numeric subfolders."
        )

    valid_subfolders.sort(key=lambda item: item[0])
    return str(valid_subfolders[-1][1])


def validate_render_dependencies(enable_render: bool) -> None:
    """Fail fast with a clear message if RLViser binary is not available."""
    if not enable_render:
        return

    # rlviser-py launches RLViser from the current working directory.
    if not Path("rlviser.exe").exists():
        raise FileNotFoundError(
            "Render requested but rlviser.exe was not found in the current working directory. "
            "Download rlviser.exe from https://github.com/VirxEC/rlviser/releases/latest "
            "and place it in the project root before running with --render."
        )


if __name__ == "__main__":
    # Read command-line options
    args = parse_args()

    # Use strategy-based naming by default or specified name if provided
    run_name = args.run_name or f"{args.strategy}_v1"
    checkpoint_load_folder = resolve_checkpoint_path(args.checkpoint_load_folder)
    validate_render_dependencies(args.render)

    # Create output folder for model checkpoints
    checkpoints_dir = Path(args.save_root) / run_name  # e.g. agents/sparse_v1/
    checkpoints_dir.mkdir(parents=True, exist_ok=True) # creates folder based on path

    metrics_logger = GoalieWandbMetricsLogger()

    # Create PPO learner with our environment builder and training settings
    # The learner will handle creating multiple worker processes, collecting experience, updating the policy, saving checkpoints, and logging metrics.
    learner = Learner(
        # On Windows, worker processes start in a way where partial(...) is more reliable than lambda.
        partial(build_env, args.strategy, enable_render=args.render),
        n_proc=args.n_proc,
        min_inference_size=max(1, int(round(args.n_proc * 0.9))),  # start updating the policy once we have at least 90% of the workers' worth of experience (improves stability by not updating too early)
        render=args.render,
        render_delay=args.render_delay,
        metrics_logger=metrics_logger,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size,
        policy_layer_sizes=(256, 256, 256),  # simple 3-layer MLP policy network with 256 nodes per layer (decides which action to take)
        critic_layer_sizes=(256, 256, 256),  # simple 3-layer MLP critic network with 256 nodes per layer (decides reward value)
        ts_per_iteration=args.ppo_batch_size,
        timestep_limit=args.timestep_limit,
        device=args.device,
        checkpoints_save_folder=str(checkpoints_dir),
        checkpoint_load_folder=checkpoint_load_folder,
        add_unix_timestamp=False, 
        save_every_ts=args.save_every_ts,
        log_to_wandb=not args.no_wandb,  
        load_wandb=not args.no_wandb,  
        wandb_run_name=run_name,
    )

    # Start training loop:
    print(f"Starting {args.strategy} training with {args.n_proc} workers on device={args.device}.")
    print(f"Training will stop at {args.timestep_limit:,} timesteps.")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")
    print(f"Resuming from checkpoint: {checkpoint_load_folder if checkpoint_load_folder else 'None (fresh run)'}")
    print(f"Rendering enabled: {args.render} (delay={args.render_delay})")
    print(f"W&B logging enabled: {not args.no_wandb}")
    learner.learn()