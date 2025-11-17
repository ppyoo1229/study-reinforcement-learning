import gymnasium as gym

# 알고리즘 불러오기
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
# 환경 생성
env = gym.make("HalfCheetah-v4") # HalfCheetah-v2라는 환경을 gymnasium 모듈을 통해 생성
# 알고리즘 생성 및 학습
sac_model = SAC('MlpPolicy', env = env, verbose=1)
sac_model.learn(1000) # 간단한 체크 위해 적은 step 사용
# 학습 결과 평가
eval_env = gym.make('HalfCheetah-v4', render_mode="human") # 성능 평가용 환경을 별도로 생성, 시각화 위해 render_mode 지정
mean_reward, std_reward = evaluate_policy(sac_model, eval_env, n_eval_episodes=5)
print("에피소드 당 평균 보상:", mean_reward, "에피소드 당 보상의 표준편차: ", std_reward)

###### 따라서 빈 칸 채우기 #####
############################################
# 1. 학습 결과(policy) 저장하기
import os
if not os.path.exists("./results"):
    os.makedirs("./results")
############################################


############################################
# 2. 저장된 policy 불러오기



############################################
# 성능 테스트 빈 칸 채우기
# 학습 결과 평가



############################################
# 3. 다운로드 받은 poicy 불러온 후 성능 테스트하기
# https://huggingface.co/sb3/sac-HalfCheetah-v3
############################################



############################################
# 성능 테스트 빈 칸 채우기



############################################