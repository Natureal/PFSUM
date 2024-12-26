import random
import math
import numpy as np
import matplotlib.pyplot as plt

def interval_generator(key, mean):

    #for occasional travelers
    if (key == "Exponential"):
        return max(1, round(np.random.exponential(mean)))

    #for commuters
    return 1

def price_generator(key, mean):

    if (key == "Normal"):
        return max(1, round(np.random.normal(mean, mean / 2)))
    elif (key == "Pareto"):
        return max(1, round(np.random.pareto(2) * mean))

    #Uniform
    return max(1, round(np.random.uniform(0, mean * 2)))

#generate a full instance of some time length
def instance_generator(length, key1, mean1, key2, mean2):

    instance = [0] * length
    idx = 0

    while (idx < length):
        price = price_generator(key2, mean2)
        instance[idx] = price

        interval = interval_generator(key1, mean1)
        idx += interval

    return instance

#generate a noisy instance from an instance
def noisy_instance_generator(instance, key2, mean2, perturb_prob):

    instance_noisy = [0] * len(instance)

    for i in range(0, len(instance)):
        instance_noisy[i] = instance[i]

        drop = (np.random.uniform(0, 1) < perturb_prob)
        add = (np.random.uniform(0, 1) < perturb_prob)
        minus = (np.random.uniform(0, 1) < perturb_prob)

        #if (drop):
         #   instance_noisy[i] = 0

        if (add):
            noise = price_generator(key2, mean2)
            instance_noisy[i] += noise
        
        #if (minus):
         #   noise = price_generator(key2, mean2)
          #  instance_noisy[i] = max(0, instance_noisy[i] - noise)
            #print("instance: {}, noisy: {}".format(instance[i], instance_noisy[i]))

    return instance_noisy

#plot an instance as a histogram
def plot_instance(instance):

    x = []
    y = []
    i = 0
    for ins in instance:
        x.append(i)
        y.append(ins)
        i += 1
        

    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.bar(x, y, width=0.5, color='C0')

    ax.grid(axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_color('C7')
    plt.show()

    return 0

#generate prediction from a instance or a noisy instance
def prediction_generator(instance, t):
	#prediction for the cost in (i, i + t)

    prediction = []
    pre_sum = [instance[0]]

    for i in range(1, len(instance)):
        v = instance[i] + pre_sum[i - 1]
        pre_sum.append(v)

    for i in range(0, len(instance)):
        prediction.append(pre_sum[min(i + t - 1, len(instance) - 1)] - pre_sum[i])

    return prediction
