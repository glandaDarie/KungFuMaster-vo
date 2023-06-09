import gym 
import numpy as np
from hyperparameters import *
from typing import Tuple

def main() -> None:
    environment : gym.Env[np.ndarray, np.ndarray | int] = gym.make("ALE/KungFuMaster-v5", render_mode="human")
    num_states : int = environment.observation_space.shape[0]
    num_actions : int = environment.action_space.n

    for episode in range(NUMBER_EPISODES):
        state : Tuple[np.ndarray, dict] = environment.reset()

        for step in range(MAX_STEPS):
            environment.render()

            action = np.random.randint(num_actions) 
            next_state, reward, _, done, info = environment.step(action)  

            if done:
                print(f"Episode {episode + 1} finished after {step + 1} steps")
                break

            state : Tuple[np.ndarray, dict] = next_state
    environment.close()

if __name__ == "__main__":
    main()
