from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl.env import ParameterSwitchEnv


def main() -> None:
    env = ParameterSwitchEnv()
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("reports/ppo_parameter_switch")


if __name__ == "__main__":
    main()
