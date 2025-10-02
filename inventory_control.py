import discrete_env
import numpy as np
import math

class Inventory(discrete_env.DiscreteEnv):
    def __init__(self):
        self.capacity = 10
        self.discount_factor = 0.99

        self.price = 20 # 제품 가격
        self.fixed_o_cost = 4 # 주문 시 고정비용
        self.o_cost = 2 # 제품 하나당 주문비용
        self.h_cost = 1 # 재고 하나, 한 번의 시간간격당 재고비용
        self.demand_rate = 3 # 한 번의 시간간격당 평균 수요량

        # DiscreteEnv 형식 맞추기
        nS = self.capacity + 1 # 가능한 상태 개수
        nA = self.capacity + 1 # 가능한 액션 개수
        isd = np.zeros(nS) # 초기 상태 분포
        isd[0] = 1.0 # 초기 재고량은 0으로 가정
        P = self._calculate_transition_prob(nS, nA)

        super(Inventory, self).__init__(nS, nA, P, isd)

        self.reset()

    def render(self, mode='human'):
        print("이번 수요량: {}, 이번 주문량: {}, 이번 수익: {}, 남은 재고량: {}".format(self.d_t, self.a_t, self.r_t,self.state))

    def close(self):
        pass

    # For VI, PI
    def _calculate_transition_prob(self, nS, nA):
        """
        :return: transition matrix
        """
        P = {s: {a: [] for a in range(nA)} for s in range(nS)} # 전이확률 저장 위한 틀; P[s][a] == [(probability, nextstate, reward, done), ...]

        for s in range(nS):
            for a in range(nA):
                if s+a <= self.capacity:
                    for s_prime in range(self.capacity + 1):
                        rew =  (s+a-s_prime) * self.price - (s+a) * self.h_cost - a * self.o_cost
                        if a > 0:
                            rew -= self.fixed_o_cost
                        if s_prime <= s+a:
                            if s_prime == 0:
                                prob = 1.0
                                for i in range(s+a-s_prime):
                                    prob -= np.power(self.demand_rate, i) * np.exp(-self.demand_rate) / math.factorial(i)
                                P[s][a].append((prob, s_prime, rew, False))
                            else:
                                d = s+a-s_prime
                                prob = np.power(self.demand_rate, d) * np.exp(-self.demand_rate) / math.factorial(d)
                                P[s][a].append((prob, s_prime, rew, False))
        return P

