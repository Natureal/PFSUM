import random
import math
import numpy as np
import matplotlib.pyplot as plt
import instance_maker as ins
import algorithms as algo


experiment_num = 1

#fixed parameters
interval_dist = "Exponential"
C = 100

#enumerating parameters
req_nums = [1000]
interval_means = [1.5]
price_means = [100]
price_dists = ["Uniform", "Exponential"]
betas = [0.8, 0.2]
Ts = [5, 10]
Lambdas = [1, 0.8, 0.6, 0.4]
replacement_rates = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
algorithms = ["sum", "fsum", "pfsum", "pdla"]
colors = {"sum": "#FFFF00", "fsum": "#FF7F00", "pfsum": "#FF0000", "pdla": "#0000FF"}

#competitive ratios
result_crs = []

first = 1

def evaluate(instance, price_dist, r_rate):
    for beta in betas:
        for T in Ts:

            offline_cost, offline_solution = algo.OFFLINE_OPTIMAL(instance, T, C, beta)
            sum_cost, sum_solution = algo.SUM(instance, T, C, beta)
            noisy_prediction = ins.prediction_generator(noisy_instance, T)
            fsum_cost, fsum_solution = algo.FSUM(instance, T, C, beta, noisy_prediction)
            pfsum_cost, pfsum_solution = algo.PFSUM(instance, T, C, beta, noisy_prediction)

            result_crs.append(("sum", beta, T, r_rate, price_dist, sum_cost / offline_cost))
            result_crs.append(("fsum", beta, T, r_rate, price_dist, fsum_cost / offline_cost))
            result_crs.append(("pfsum", beta, T, r_rate, price_dist, pfsum_cost / offline_cost))

            for Lambda in Lambdas:
                pdla_cost, pdla_solution = algo.PDLA_FOR_BAHNCARD(instance, T, C, beta, Lambda, noisy_prediction)
                result_crs.append(("pdla", beta, T, Lambda, r_rate, price_dist, pdla_cost / offline_cost))

for rn in req_nums:
    for interval_mean in interval_means:
        for price_mean in price_means:
            for price_dist in price_dists:
                for r_rate in replacement_rates:
                    for exp_count in range(0, experiment_num):

                        #generate an instance
                        np.random.seed(exp_count)
                        instance = ins.instance_generator(interval_mean, price_mean, rn, interval_dist, price_dist)
                        noisy_instance = ins.noisy_instance_generator(instance, price_mean, price_dist, r_rate)
                        #if (first == 1):
                        #    ins.plot_instance(instance)
                        #    first = 0
                        evaluate(instance, price_dist, r_rate)

#differentiate results using price distribution
result_price_dist_crs = {"Uniform": [0, 0], "Exponential": [0, 0]}
for r in result_crs:
    if (r[0] == "pdla"):
        result_price_dist_crs[r[5]][0] += r[6]
        result_price_dist_crs[r[5]][1] += 1
    else:
        result_price_dist_crs[r[4]][0] += r[5]
        result_price_dist_crs[r[4]][1] += 1

print(result_price_dist_crs["Uniform"][0] / result_price_dist_crs["Uniform"][1])
print(result_price_dist_crs["Exponential"][0] / result_price_dist_crs["Exponential"][1])

#fix beta = 0.8, T = 10, price_distribution = Uniform
points = {"sum": {}, "fsum": {}, "pfsum": {}, "pdla": {}}
for r in result_crs:
    if (r[0] == "pdla"):
        if (r[1] != 0.8 or r[2] != 10 or r[5] != "Uniform"): continue
        points["pdla"][r[4]] = r[6]
        
    else:
        if (r[1] != 0.8 or r[2] != 10 or r[4] != "Uniform"): continue
        points[r[0]][r[3]] = r[5]

for algo in algorithms:
    l = []
    for r_rate in replacement_rates:
        print(algo, ", ", r_rate)
        print(points[algo][r_rate])
        l.append(points[algo][r_rate])
    plt.plot(replacement_rates, l, color=colors[algo], label=algo)

plt.xlabel("Replacement rate")
plt.ylabel("Competitive ratio")
plt.legend(loc=1, prop={'size': 15}, bbox_to_anchor=(1.01,0.5))
plt.gcf().subplots_adjust(left=0.14, bottom=0.14)
plt.show()

