# 2. 설치된 stable_baselines3 불러오기
import stable_baselines3
# 3. stable_baselines3에서 사용 할 강화학습 알고리즘 (PPO) 불러오기
from stable_baselines3 import PPO
# 4. openAI gym의 강화학습 환경을 사용하기 위해 gym 불러오기
import gym # 주의: gymnasium이 아니라 gym 불러옵니다.
######## 물류센터 모델 완성본 ########
from gym import spaces
import numpy as np
from gym.utils import seeding # random seed control 위해 import

class WareHouse(gym.Env):
    def __init__(self, map, lot_max, arrvial_rate):
        self.map = map
        self.lot_max = lot_max
        self.height, self.width = map.shape
        self.depot = None
        self.stations = []

        for c in range(self.width):
            for r in range(self.height):
                if map[r, c] == 'D':
                    self.depot = (r, c)
                elif map[r, c] == 'S':
                    self.stations.append((r, c))

        self.action_space = spaces.Discrete(4) # LEFT, UP, RIGHT, DOWN
        self.observation_space = spaces.MultiDiscrete([self.height, self.width, 2] + [self.lot_max+1]*len(self.stations))
        # location y, x, load or not, lot per stations

        self.arrvial_rate = arrvial_rate # per time unit
        self.c_loss = 0.1 # loss cost per lot
        self.reward_per_success = 10.0

        self.viewer = None

        self.reset()

    def arrival(self):
        """
        :return number of newly arrived lots per station
        Assume poisson distribution with arrival rate
        """
        return np.random.poisson(self.arrvial_rate, len(self.stations))

    def get_cost(self, lot_per_stations):
        """
        :return holding cost
        Compute cost given the current number of lots
        Assume coefficient*x^2
        """
        cost = 0.0
        coeff = 0.01
        for x in lot_per_stations:
            cost += coeff*(x**2)
        return cost

    def step(self, action):
        row, col, load, *lot_per_stations = self.state
        prev_row, prev_col = row, col
        reward = -self.get_cost(lot_per_stations)
        # print("Holding costs: ", reward)
        done = False
        info = {}

        # New lot arrival
        arrv = self.arrival()
        # print("New arrivals: ", arrv)
        lot_per_stations = np.array(lot_per_stations) + arrv

        # Move
        if action == 0: # LEFT
            col -= 1
        elif action == 1: # UP
            row -= 1
        elif action == 2: # RIGHT
            col += 1
        elif action == 3: # DOWN
            row += 1
        else:
            raise Exception('bad action {}'.format(action))

        c_loc = (row, col)
        blocked = False
        if col < 0 or col >= self.width or row < 0 or row >= self.height: # out of bounds, cannot move
            blocked = True
        elif c_loc in self.stations:
            if lot_per_stations[self.stations.index(c_loc)] > 0:
                if load: # blocked at a station
                    blocked = True
                else: # Take a lot
                    load = 1
                    lot_per_stations[self.stations.index(c_loc)] -= 1
        elif c_loc == self.depot: # into a goal cell
            if load:
                load = 0
                reward += self.reward_per_success
                # print("Success move")
                info['success'] = True

        # Set max lot_per_stations
        for i in range(len(lot_per_stations)):
            if lot_per_stations[i] > self.lot_max:
                reward -= (lot_per_stations[i] - self.lot_max) * self.c_loss # Compute loss cost
                lot_per_stations[i] = self.lot_max
        # lot_per_stations = [(self.lot_max if x > self.lot_max else x) for x in lot_per_stations]

        if blocked: # cannot move
            self.state = (prev_row, prev_col, load,) + tuple(lot_per_stations)
            return np.array(self.state), reward, done, info
        else:
            self.state = c_loc + (load,) + tuple(lot_per_stations)
            return np.array(self.state), reward, done, info

    def reset(self):
        # Initial state is depot location without any lots at stations
        self.state = self.depot + (0,) * (len(self.stations)+1)
        return np.array(self.state)  # reward, done, info can't be included

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

######## Warehouse 클래스를 강화학습 알고리즘에 적용하기 ########
MAP = [
    [' ', ' ', 'D', ' ', ' '],
    [' ', 'S', ' ', 'S', ' '],
    [' ', ' ', ' ', ' ', ' '],
    [' ', 'S', ' ', 'S', ' '],
] # D는 물건을 가져다 놓을 depot, S는 물류가 들어오는 Station
action_name = {0:"LEFT", 1:"UP", 2:"RIGHT", 3:"DOWN"}
env = WareHouse(np.array(MAP),3,0.06) # 물류센터 지도, 스테이션 버퍼 사이즈, 평균적인 물건 도착율을 넣어줌

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=int(1.0e5))

test_env = WareHouse(np.array(MAP),3,0.02)

obs = test_env.reset()
count_success = 0 # 물건 옮기기 성공
cumul_reward = 0
render = False

for i in range(1000):
    if not render:
      print("state: ", np.array(test_env.state))
    action, _states = model.predict(obs)
    obs, rewards, dones, info = test_env.step(action)

    cumul_reward += rewards
    if info.get('success'):
        count_success += 1
    if not render:
      print("action: ", action_name[int(action)])
      print("reward this step: ", rewards)
      print("total reward: ", cumul_reward)
      print("="*50)

print("Total successful move: ", count_success)
test_env.close()