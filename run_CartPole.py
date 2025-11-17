# 환경 불러오기
import gymnasium as gym

# 알고리즘 불러오기
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# 환경 생성
cartpole_env = gym.make('CartPole-v1')
# 알고리즘 생성 및 학습
ppo_model = PPO('MlpPolicy', cartpole_env, verbose=1)
ppo_model.learn(100000)
# 학습 결과 평가
eval_env = gym.make('CartPole-v1', render_mode="human") # 성능 평가용 환경을 별도로 생성, 시각화 위해 render_mode 지정
mean_reward, std_reward = evaluate_policy(ppo_model, eval_env, n_eval_episodes=5)
print("에피소드 당 평균 보상:", mean_reward, "에피소드 당 보상의 표준편차: ", std_reward)


