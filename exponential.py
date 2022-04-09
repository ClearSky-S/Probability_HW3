import numpy as np
import matplotlib.pyplot as plt


def exponential(n=1000, p=0.05, seed=0, time_limit=1000):
    result = 0
    np.random.seed(seed=seed)
    while True:
        if result >= n*time_limit:
            break
        if np.random.uniform(low=0, high=1) < p:
            break
        result += 1
    return result/n


def pdf_exp(t, lamda):
    return lamda*np.exp(-lamda*t)

def cdf_exp(t, lam):
    return 1-np.exp(-lamda*t)

n = 1000  # 단위 시간 동안의 시행 횟수 (총 시행 횟수가 아니다)
# p = 0.05
p = float(input("input p(0~1) >>> "))
# seed = 0
seed = int(input("input seed(integer) >>> "))
lamda = n * p
time_limit = 2  # 무한 번 반복 시행 하지 않도록 하는 시간 제한

trial = 1000
exp_simulation_result = np.zeros(n*time_limit)

for i in range(0, trial):
    x = exponential(n, p=p, seed=seed+i)
    if(x < time_limit ):
        exp_simulation_result[int(x*1000)] += 1

for i in range(0, n*time_limit):
    exp_simulation_result[i] /= trial

exp_math_result = np.array([pdf_exp(k/n, lamda)/n for k in range(0, n*time_limit)])


plt.title('exp pdf')
plt.xlabel('x')
plt.ylabel('f (X=x)')
plt.xlim([0, 100])
plt.plot(exp_simulation_result, label="Simulation")
plt.plot(exp_math_result, label="Theory")
plt.legend()
plt.show()

for i in range(101):
    exp_simulation_result[i+1] = exp_simulation_result[i] + exp_simulation_result[i+1]
exp_math_result = np.array([cdf_exp(k/n, lamda) for k in range(0, n*time_limit)])

plt.title('exp cdf')
plt.xlabel('T')
plt.ylabel('P (X<T)')
plt.xlim([0, 100])
plt.plot(exp_simulation_result, label="Simulation")
plt.plot(exp_math_result, label="Theory")
plt.legend()
plt.show()