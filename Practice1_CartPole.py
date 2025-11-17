# Copy of Practice1_CartPole.ipynb
##############################################################################
# 2. 설치된 stable_baselines3 불러오기
import stable_baselines3
# 비디오 출력 위한 클래스 추가 불러오기
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
##############################################################################
# 3. stable_baselines3에서 사용 할 강화학습 알고리즘 (DQN) 불러오기
from stable_baselines3 import DQN
##############################################################################
# 4. openAI gym의 강화학습 환경을 사용하기 위해 gym 불러오기
import gymnasium as gym
##############################################################################
# 5. gym을 통해 cartpole 환경 생성하기
cartpole_env = gym.make('CartPole-v1')
##############################################################################
# 6. Cartpole 환경에 대한 정보 확인
# 2 종류의 액션: (좌, 우)
# 4 차원의 연속적인 상태 공간:
# 카트의 위치: -4.8 ~ 4.8
# 카트의 속력: -Inf ~ Inf
# 막대기의 각도: -0.418 rad (-24도) ~ 0.418 rad (24도)
# 막대기 끝 부분의 속도: -Inf, Inf
# 한 에피소드 당 최대 길이: 500
print("Action space: ", cartpole_env.action_space)
print("Observation space: ", cartpole_env.observation_space)
print("Maximum episode steps: ", cartpole_env.spec.max_episode_steps)
##############################################################################
# 7. DQN 에이전트 (DQN 강화학습 모델)을 hyperparameter를 지정하여 생성하기
# Cart-pole 환경은 이미지 데이터를 사용하는 환경이 아니기 때문에 CNN이 아닌 fully-connected layer ('MlpPolicy') 사용
# 다른 hyperparameter들은 stable-baselines3 zoo에서 제공하는 값을 참고하여 사용 (https://github.com/DLR-RM/rl-baselines3-zoo)
hyper_param = {'n_timesteps': 5e4,
               'policy': 'MlpPolicy',
               'learning_rate': 2.3e-3,
               'batch_size': 64, # MSE 계산에 사용 할 샘플 개수
               'buffer_size': 100000, # replay buffer에 저장 할 샘플 (s,a,r,s') 수
               'learning_starts': 1000,
               'gamma': 0.99,
               'target_update_interval': 10, # Target network에 학습된 parameter를 복사해 넣는 주기
               'train_freq': 256, # Q-network를 몇 스텝마다 학습할 것인지 c.f.) 원래 DQN은 1 step
               'gradient_steps': 128, # MSE 계산을 몇 번 할 것인지. c.f.) 원래 DQN은 1번
               'exploration_fraction': 0.16, # 전체 학습 기간 중 탐색을 위해 큰 epsilon 값을 감소시키는 기간. 이 기간 이후로는 수렴된 epsilon 값 사용
               'exploration_final_eps': 0.04, # 고정적으로 사용할 epsilon 값
               'policy_kwargs': dict(net_arch=[256, 256]),
               'verbose': 1 # 학습 진행 상황 출력하고 싶을 때 입력
               }
dqn_model = DQN(policy = hyper_param['policy'],
                env = cartpole_env,
                learning_rate = hyper_param['learning_rate'],
                batch_size = hyper_param['batch_size'],
                buffer_size = hyper_param['buffer_size'],
                learning_starts = hyper_param['learning_starts'],
                gamma = hyper_param['gamma'],
                target_update_interval = hyper_param['target_update_interval'],
                train_freq = hyper_param['train_freq'],
                gradient_steps = hyper_param['gradient_steps'],
                exploration_fraction = hyper_param['exploration_fraction'],
                exploration_final_eps = hyper_param['exploration_final_eps'],
                verbose = hyper_param['verbose'],
                policy_kwargs = hyper_param['policy_kwargs']
                )
##############################################################################
# 8. 학습을 시작하기 전에 DQN 에이전트의 성능 확인하기
# stable-baselines의 evaluate_policy(RL 모델, 평가 환경, 평가에 사용할 에피소드 수) 함수 사용.
# Warning의 의미: Monitor 클래스를 통해 gym 환경의 에피소드 종료 시그널(done)이 실제 에피소드 종료를 의미하도록 만드는 것. 일부 환경에서는 학습을 위해 실제 종료 이전에 중간 구분 지점을 잡기도 하기 때문.
from stable_baselines3.common.evaluation import evaluate_policy
eval_env = gym.make('CartPole-v1') # 성능 평가용 환경을 별도로 생성
mean_reward, std_reward = evaluate_policy(dqn_model, eval_env, n_eval_episodes=100) # episode 100개 생성하여 평가
print("에피소드 당 평균 보상:", mean_reward, "에피소드 당 보상의 표준편차: ", std_reward)
##############################################################################
# 9. DQN 에이전트 학습시키기
dqn_model.learn(total_timesteps=hyper_param['n_timesteps'])
##############################################################################
# 10. 학습된 결과 확인하기
mean_reward, std_reward = evaluate_policy(dqn_model, eval_env, n_eval_episodes=100)
print("에피소드 당 평균 보상:", mean_reward, "에피소드 당 보상의 표준편차: ", std_reward)
##############################################################################
# 실습: Proximal Policy Optimization (PPO) 알고리즘을 이용해서 cart-pole 문제 풀기
from stable_baselines3 import PPO

# 채워넣어보기 실습
# 1. Cart-pole 환경 만들기

# 2. PPO 에이전트 만들기; policy 파라미터('MlpPolicy')와 학습할 환경만 넘겨주기

# 3. 학습 시작! 총 100000 step만큼 진행.

# 학습 결과 확인하기

##############################################################################
# 11. Video를 통해 랜덤 에이전트와 학습된 에이전트가 cart-pole 환경에서 움직이는 것 비교하기
# 시각화 안될 시, anaconda prompt에서 "pip install pyglet", "pip install pygame" 실행을 통해 pyglet, pygame 설치

# 1. 랜덤 에이전트
test_env = gym.make('CartPole-v1', render_mode="human") # 성능 평가용 환경을 별도로 생성, 시각화 위해 render_mode 지정

observation = test_env.reset() # 환경 초기화. observation은 첫 관측값
cum_reward = 0 # 에피소드 끝날 때까지의 총 보상값
done = False # 에피소드 종료 조건 저장
while not done: # 에피소드 종료 때까지 실행
    test_env.render() # gym의 render 함수는 gym 환경을 시각화하기 위한 용도로 사용

    action = test_env.action_space.sample() # 환경명.action_space.sample()은 가능한 액션을 랜덤하게 하나 선택
    observation, reward, done, truncated, info = test_env.step(action)  # step 함수를 통해 선택된 액션을 수행하고, (다음 관측값, 보상, 에피소드 종료 여부, 부가적인 정보)를 얻음
    cum_reward += reward

print("이번 에피소드의 총 보상값: ", cum_reward)
test_env.close()

# 2. 학습된 에이전트 (DQN)
test_env = gym.make('CartPole-v1', render_mode="human") # 성능 평가용 환경을 별도로 생성, 시각화 위해 render_mode 지정

observation = test_env.reset() # 환경 초기화. observation은 첫 관측값
cum_reward = 0 # 에피소드 끝날 때까지의 총 보상값
done = False # 에피소드 종료 조건 저장
while not done: # 에피소드 종료 때까지 실행
    test_env.render() # gym의 render 함수는 gym 환경을 시각화하기 위한 용도로 사용

    action, _ = dqn_model.predict(observation) # 학습된 에이전트로 부터 액션 선택
    observation, reward, done, truncated, info = test_env.step(action)  # step 함수를 통해 선택된 액션을 수행하고, (다음 관측값, 보상, 에피소드 종료 여부, 부가적인 정보)를 얻음
    cum_reward += reward

print("이번 에피소드의 총 보상값: ", cum_reward)
test_env.close()
##############################################################################