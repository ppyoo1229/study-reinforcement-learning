# run_VIandPI.py
from VIandPI import Value_iteration, Policy_iteration
from inventory_control import Inventory

# 환경 생성
env = Inventory()

# VI
print("=" * 50)
print("Value Iteration")
print("=" * 50)
vi_model = Value_iteration(env)
policy, value = vi_model.solve(max_iter=10000)
print("Policy: ", policy)
print("Value: ", value)
# PI
print("=" * 50)
print("Policy Iteration")
print("=" * 50)
pi_model = Policy_iteration(env)
policy, value = pi_model.solve(max_iter=10000)
print("Policy: ", policy)
print("Value: ", value) 
