import gym
import gym_minigrid
import time
import random
import gym_test
from stable_baselines3 import DQN


env_name = 'test-v0'
env = gym.make(env_name) # charge environment

print(env.observation_space)
print(env.action_space)


class Agent():
    def init(self, env) -> None:
        self.action_size = env.action_space.n
        print("action size:", self.action_size)

    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        action = random.choice(range(self.actionsize))
        return action

agent = Agent()
state = env.reset()



model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

model.save("dqn_tp")

del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_tp")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

# for _ in range(20000):
#     action = env.action_space.sample()
#     # action = agent.get_action(state)
#     state, reward, done, info =  env.step(action)
#     env.render()
#     if done:
#         env.reset()
# env.close()