import numpy as np
import matplotlib.pyplot as plt


def poi(n=1000, p=0.05, seed=0):
    result = 0
    np.random.seed(seed=seed)
    for i in range(0, n):
        if np.random.uniform(low=0, high=1) < p:
            result += 1
    return result


def p_poi(k, lam):
    return (lam**k)*np.exp(-lam)/np.math.factorial(k)


trial = 1000  # should be 1000 at actual run
n = 1000
# p = 0.05
p = float(input("input p(0~1) >>> "))
# seed = 0
seed = int(input("input seed(integer) >>> "))
lamda = n * p

poi_simulation_result = np.zeros(int(lamda * 2))
for i in range(0, trial):
    x = poi(n, p=p, seed=seed+i)
    if(x < int(lamda * 2)):
        poi_simulation_result[x] += 1

for i in range(int(lamda * 2)):
    poi_simulation_result[i] = poi_simulation_result[i]/trial

# print(poi_simulation_result)
# print(poi_simulation_result[int(lamda)])

poi_math_result = np.array([p_poi(k, lamda) for k in range(int(lamda * 2))])
# print(poi_math_result)
# print(poi_math_result[int(lamda)])

plt.title('Poisson Distribution')
plt.xlabel('x')
plt.ylabel('P (X=x)')
plt.plot(poi_simulation_result, label="simulation")
plt.plot(poi_math_result, label="math")
plt.legend()
plt.show()

mean = 0
for i in range(int(lamda * 2)):
    mean += i*poi_math_result[i]
print(mean)