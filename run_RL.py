from windy_gridworld_edited import WindyGridworldEnv
from basic_RL_alg import qlearning, sarsa
import time

if __name__ == '__main__':
    env = WindyGridworldEnv()

    # Random agent (before learning)
    print("=" * 50)
    print("Random agent")
    obs = env.reset()
    for i in range(20): # 최대 step 수로 확인하고 싶을 때.
        dones = False
        env.render()
        action = env.action_space.sample()
        obs, rewards, dones, info = env.step(action)
        time.sleep(1)

    print("="*50)


    # Q-learning
    print("Q-learning agent")
    model = qlearning(env, gamma=0.99, learning_rate=0.5, eps=0.1)
    model.learn(int(2.0e4))

    # # SARSA
    # model = sarsa(env, gamma=0.99, learning_rate=0.5, eps=0.1)
    # model.learn(int(3.0e4))

    obs = env.reset()
    for episodes in range(3): # 최대 episode 종료 횟수로 확인하고 싶을 때.
    # for i in range(1000): # 최대 step 수로 확인하고 싶을 때.
        dones = False
        episode_reward = 0
        while not dones:
            # action = env.action_space.sample()
            env.render()
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            time.sleep(1)

            episode_reward += rewards
        print("이번 에피소드 보상합: ", episode_reward)
        print("="*50)

    print(model.getQvalues())