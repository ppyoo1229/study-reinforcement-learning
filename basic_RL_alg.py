import numpy as np

class qlearning():
    # Simple Q-learning

    def __init__(self, env, gamma=0.99, learning_rate=0.05, eps=0.1):
        self.env = env
        self.gamma = gamma
        self.l_rate = learning_rate
        self.eps = eps

        self.num_states = env.observation_space.n # Discrete states
        self.num_actions = env.action_space.n # Discrete actions

        self.q_table = np.zeros((self.num_states, self.num_actions))
        # self.q_table = np.ones((self.num_states, self.num_actions)) * 5000

    def learn(self, total_timesteps):
        obs = self.env.reset()
        for step in range(total_timesteps):
            if (step+1)%5000 == 0:
                print("steps:" , step + 1)
            action = self.eps_greedy_action(obs)
            n_obs, reward, done, info = self.env.step(action)

            if done:
                q_target = reward
            else:
                q_target = reward + self.gamma * np.max(self.q_table[n_obs])
            self.q_table[obs, action] = (1 - self.l_rate) * self.q_table[obs, action] + self.l_rate * q_target
            obs = n_obs
            if done:
                obs = self.env.reset()

    def predict(self, observation, deterministic = True):
        states = None
        if deterministic:
            return np.argmax(self.q_table[observation]), states
        else:
            return self.eps_greedy_action(observation), states

    def eps_greedy_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def getQvalues(self):
        return self.q_table # q-value

    def getPolicy(self):
        return np.argmax(self.q_table, axis=1)

class sarsa():
    # Simple SARSA

    def __init__(self, env, gamma=0.99, learning_rate=0.05, eps=0.1):
        self.env = env
        self.gamma = gamma
        self.l_rate = learning_rate
        self.eps = eps

        self.num_states = env.observation_space.n # Discrete states
        self.num_actions = env.action_space.n # Discrete actions

        self.q_table = np.zeros((self.num_states, self.num_actions))

    def learn(self, total_timesteps):
        obs = self.env.reset()
        action = self.eps_greedy_action(obs)
        for _ in range(total_timesteps):
            n_obs, reward, done, info = self.env.step(action)
            next_action = self.eps_greedy_action(n_obs)

            if done:
                q_target = reward
            else:
                q_target = reward + self.gamma * self.q_table[n_obs, next_action]
            self.q_table[obs, action] = (1 - self.l_rate) * self.q_table[obs, action] + self.l_rate * q_target
            # self.q_table[obs, action] = self.q_table[obs, action] + self.l_rate * (q_target - self.q_table[obs, action])
            obs = n_obs
            action = next_action
            if done:
                obs = self.env.reset()
                action = self.eps_greedy_action(obs)

    def predict(self, observation, deterministic = True):
        states = None
        if deterministic:
            return np.argmax(self.q_table[observation]), states
        else:
            return self.eps_greedy_action(observation), states

    def eps_greedy_action(self, observation):
        if np.random.random() < self.eps:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def getQvalues(self):
        return self.q_table # q-value

    def getPolicy(self):
        return np.argmax(self.q_table, axis=1)