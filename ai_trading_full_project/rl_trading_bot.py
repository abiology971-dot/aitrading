import warnings

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """

    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        # Trading state
        self.balance = initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: [balance, shares_held, cost_basis, current_price, volume, position_value, total_value]
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        return self._get_observation()

    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1

        current_price = self.data.iloc[self.current_step]["Close"]
        volume = self.data.iloc[self.current_step]["Volume"]

        position_value = self.shares_held * current_price
        total_value = self.balance + position_value

        obs = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.shares_held / 100,  # Normalized shares
                self.cost_basis / 1000 if self.cost_basis > 0 else 0,
                current_price / 100,  # Normalized price
                volume / 1e8,  # Normalized volume
                position_value / self.initial_balance,
                total_value / self.initial_balance,
            ],
            dtype=np.float32,
        )

        return obs

    def step(self, action):
        """Execute one step in the environment"""
        current_price = self.data.iloc[self.current_step]["Close"]
        prev_value = self.balance + (self.shares_held * current_price)

        # Execute action
        if action == 1:  # Buy
            # Buy 10 shares if we have enough balance
            shares_to_buy = 10
            total_cost = shares_to_buy * current_price

            if self.balance >= total_cost:
                self.balance -= total_cost
                self.cost_basis = (
                    (self.cost_basis * self.shares_held) + total_cost
                ) / (self.shares_held + shares_to_buy)
                self.shares_held += shares_to_buy

        elif action == 2:  # Sell
            # Sell all shares if we have any
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.total_shares_sold += self.shares_held
                self.total_sales_value += self.shares_held * current_price
                self.shares_held = 0
                self.cost_basis = 0

        # Move to next step
        self.current_step += 1

        # Calculate reward
        current_value = self.balance + (self.shares_held * current_price)
        reward = (
            current_value - prev_value
        ) / self.initial_balance  # Normalized reward

        # Check if done
        done = self.current_step >= len(self.data) - 1

        # Get new observation
        obs = self._get_observation()

        # Additional info
        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "total_value": current_value,
            "profit": current_value - self.initial_balance,
        }

        return obs, reward, done, info

    def render(self, mode="human"):
        """Render the environment"""
        current_price = self.data.iloc[self.current_step]["Close"]
        current_value = self.balance + (self.shares_held * current_price)
        profit = current_value - self.initial_balance

        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Current price: ${current_price:.2f}")
        print(f"Total value: ${current_value:.2f}")
        print(f"Profit: ${profit:.2f} ({(profit / self.initial_balance) * 100:.2f}%)")
        print("-" * 50)


def train_rl_agent(timesteps=5000, save_path="rl_trading_bot"):
    """
    Train the RL trading agent with optimized settings
    """
    print("Loading trading data...")
    data = pd.read_csv("stock_data.csv")

    # Check if required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Missing columns. Available: {data.columns.tolist()}")
        return None

    print(f"Data loaded: {len(data)} rows")

    # Split data for training and testing
    train_size = int(0.8 * len(data))
    train_data = data[required_cols].iloc[:train_size]
    test_data = data[required_cols].iloc[train_size:]

    print(f"Training data: {len(train_data)} rows")
    print(f"Testing data: {len(test_data)} rows")

    # Create environments
    print("\nCreating trading environment...")
    train_env = DummyVecEnv([lambda: TradingEnv(train_data)])
    eval_env = DummyVecEnv([lambda: TradingEnv(test_data)])

    # Create PPO agent with optimized hyperparameters
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # Train the agent
    print(f"\nTraining agent for {timesteps} timesteps...")
    print("This may take a few minutes...\n")

    try:
        model.learn(total_timesteps=timesteps, progress_bar=True)

        # Save the model
        model.save(save_path)
        print(f"\nModel saved to {save_path}")

        # Evaluate on test data
        print("\nEvaluating on test data...")
        obs = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < len(test_data):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            steps += 1

        print(f"Test episode completed in {steps} steps")
        print(f"Total reward: {total_reward[0]:.4f}")
        print(f"Final balance: ${info[0]['balance']:.2f}")
        print(f"Final total value: ${info[0]['total_value']:.2f}")
        print(f"Final profit: ${info[0]['profit']:.2f}")

        return model

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save(save_path + "_interrupted")
        print(f"Model saved to {save_path}_interrupted")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_agent(model_path="rl_trading_bot", data_path="stock_data.csv"):
    """
    Test a trained RL agent
    """
    try:
        print("Loading trained model...")
        model = PPO.load(model_path)

        print("Loading test data...")
        data = pd.read_csv(data_path)
        test_data = data[["Open", "High", "Low", "Close", "Volume"]].iloc[
            -500:
        ]  # Last 500 days

        print(f"Testing on {len(test_data)} days...")

        # Create test environment
        env = TradingEnv(test_data)

        obs = env.reset()
        done = False
        total_reward = 0
        actions_taken = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            actions_taken[action] += 1

        print("\n" + "=" * 50)
        print("Testing Results:")
        print("=" * 50)
        print(f"Initial balance: $10,000.00")
        print(f"Final balance: ${info['balance']:.2f}")
        print(f"Final shares held: {info['shares_held']}")
        print(f"Final total value: ${info['total_value']:.2f}")
        print(f"Total profit: ${info['profit']:.2f}")
        print(f"Return: {(info['profit'] / 10000) * 100:.2f}%")
        print(f"\nActions taken:")
        print(f"  Hold: {actions_taken[0]}")
        print(f"  Buy: {actions_taken[1]}")
        print(f"  Sell: {actions_taken[2]}")
        print("=" * 50)

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Reinforcement Learning Trading Bot")
    print("=" * 60)

    # Train the agent (reduced timesteps for faster training)
    model = train_rl_agent(timesteps=3000, save_path="rl_trading_bot")

    if model is not None:
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)

        # Test the trained agent
        print("\nTesting the trained agent...")
        test_agent("rl_trading_bot")
    else:
        print("\nTraining failed!")
