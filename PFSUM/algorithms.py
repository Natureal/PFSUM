import math

#5 algorithms: (1)SUM, (2)FSUM, (3)PFSUM, (4)PDLA, (5)Offline Optimal Dynamic Programming

#SUM purchases a bahncard at time t whenever the T-recent-regular-cost is at least gamma.
def SUM(instance, T, C, beta):
    length = len(instance)
    if (length == 0):
        return (0, [])

    regular_cost = [0] * length
    gamma = C / (1 - beta)
    cost = 0
    solution = []
    T_recent_regular_cost = 0
    last_buy_time = -T
    
    for i in range(0, length):
        if (i - T >= 0):
            T_recent_regular_cost -= regular_cost[i - T]

        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
            if (T_recent_regular_cost + instance[i] >= gamma):
                #buy a bahncard
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
            else:
                cost += instance[i]
                T_recent_regular_cost += instance[i]
                regular_cost[i] = instance[i]

    return (cost, solution)

#SUM_W purchases a bahncard at time t whenever the sum of (T-w)-recent-regular-cost and w-future-cost is at least gamma.
def SUM_W(instance, T, C, beta, prediction_w):
	length = len(instance)
	if (length == 0):
		return (0, [])

	w = int(T / 2)
	regular_cost = [0] * length
	gamma = C / (1 - beta)
	cost = 0
	solution = []
	T_w_recent_regular_cost = 0
	last_buy_time = -T

	for i in range(0, length):
		if (i - T + w >= 0):
			T_w_recent_regular_cost -= regular_cost[i - T + w]

		if (last_buy_time + T - 1 >= i):
			cost += beta * instance[i]
		else:
			if (T_w_recent_regular_cost + instance[i] + prediction_w[i] >= gamma):
				#buy a bahncard
				cost += C + beta * instance[i]
				last_buy_time = i
				solution.append(i)
			else:
				cost += instance[i]
				T_w_recent_regular_cost += instance[i]
				regular_cost[i] = instance[i]

	return (cost, solution)

#FSUM purchases a bahncard at time t whenever the predicted T-future-cost is at least gamma.
def FSUM(instance, T, C, beta, prediction):
    #length = len(instance)
    #if (length == 0):
    #    return (0, [])

    #gamma = C / (1 - beta)
    #cost = 0
    #solution = []
    #last_buy_time = -T

    #for i in range(0, length):
    #    if (last_buy_time + T - 1 >= i):
    #        cost += beta * instance[i]
    #    else:
	#		#prediction[i] means the predicted cost in (i, i + T)
    #        if (instance[i] + prediction[i] >= gamma):
    #            #buy a bahncard
    #            cost += C + beta * instance[i]
    #            last_buy_time = i
    #            solution.append(i)
    #        else:
    #            cost += instance[i]

    #return (cost, solution)
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    cost = 0
    solution = []
    last_buy_time = -T
    T_recent_cost = 0
    regular_cost = [0] * length
    T_recent_regular_cost = 0
    future_threshold = (math.sqrt(1 + 4 * beta) - 1 + 2 * beta) / (2 * beta) * gamma

    for i in range(0, length):
        if (i - T >= 0):
            T_recent_cost -= instance[i - T]
            T_recent_regular_cost -= regular_cost[i - T]

        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
			#prediction[i] means the predicted cost in (i, i + T)
            if ((T_recent_cost + instance[i] >= future_threshold and instance[i] + prediction[i] >= gamma) or (T_recent_regular_cost + instance[i] >= 2 * gamma)):
                #buy a bahncard
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
                print("bought by fsum'''")
            else:
                cost += instance[i]
                T_recent_regular_cost += instance[i]
                regular_cost[i] = instance[i]

        T_recent_cost += instance[i]
            
    return (cost, solution)

#Ski-rental_Lambda.
def SRL(instance, T, C, beta, prediction, predicted_instance, Lambda):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    cost = 0
    solution = []
    last_buy_time = -T
    pre_sum = [instance[0]]
    for i in range(1, length):
        pre_sum.append(instance[i] + pre_sum[i - 1])

    for i in range(0, length):
        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
            buy_card = 0
            for j in range(max(0, i - T + 1), i + 1):
                past_sum = pre_sum[i]
                if j > 0:
                    past_sum -= pre_sum[j - 1]
                if (instance[j] + prediction[j] >= gamma and past_sum >= Lambda * gamma):
                    buy_card = 1
                    break
                elif (past_sum >= gamma / Lambda):
                    buy_card = 1
                    break

            if (buy_card == 1):
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
            else:
                cost += instance[i]

    return (cost, solution)
    #for i in range(0, T):
    #    instance.append(0)
    #    predicted_instance.append(0)
    #    prediction.append(0)
    
#    pre_sum = [instance[0]]
#    for i in range(1, len(instance)):
#        pre_sum.append(pre_sum[i - 1] + instance[i])
#
#    npre_sum = [predicted_instance[0]]
#    for i in range(1, len(instance)):
#        npre_sum.append(npre_sum[i - 1] + predicted_instance[i])
#    
#    dp = []
#    N = []
#    pdp = [0] * len(instance)
#    pN = [0] * len(instance)
#
#    if instance[0] > C + beta * instance[0]:
#        dp.append(C + beta * instance[0])
#        N.append(1)
#    else:
#        dp.append(instance[0])
#        N.append(0)
#
#    for i in range(1, len(instance)):
#        dp.append(math.inf)
#        N.append(0)
#
#        if (i - T < 0):
#            if C + beta * pre_sum[i] < dp[i]:
#                dp[i] = C + beta * pre_sum[i]
#                N[i] = 1
#        else:
#            if dp[i - T] + C + beta * (pre_sum[i] - pre_sum[i - T]) < dp[i]:
#                dp[i] = dp[i - T] + C + beta * (pre_sum[i] - pre_sum[i - T])
#                N[i] = N[i - T] + 1
#
#        if dp[i - 1] + instance[i] < dp[i]:
#            dp[i] = dp[i - 1] + instance[i]
#            N[i] = N[i - 1]
#
#    if Lambda == 0.6:
#        regular_cost = [0] * len(instance)
#        gamma = C / (1 - beta)
#        past_th = beta * gamma
##future_th = gamma
#        future_th = (math.sqrt(1 + 4 * beta) - 1 + 2 * beta) / (2 * beta) * gamma
#        cost = 0
#        solution = []
#        last_buy_time = -T
#        last_flag = 0
#        T_recent_cost = 0
#        T_recent_regular_cost = 0
#
#        for i in range(0, len(instance) - T):
#            if (i - T >= 0):
#                T_recent_regular_cost -= regular_cost[i - T]
#                T_recent_cost -= instance[i - T]
#
#            if (last_buy_time + T - 1 >= i):
#                cost += beta * instance[i]
#            else:
#	    		#prediction[i] means the predicted cost in (i, i + T)
#                #if (instance[i] >= (C - T_recent_cost * (1.0 - beta)) / (2.0 * (1 - beta)) and instance[i] + prediction[i] >= gamma):
#                #if (instance[i] >= (C - T_recent_regular_cost * (1.0 - beta)) / (2.0 * (1 - beta)) and instance[i] + prediction[i] >= gamma):
#                #if (instance[i] + prediction[i] >= gamma and T_recent_cost + 2 * instance[i] + prediction[i] >= 2 * gamma):
#                if (instance[i] + prediction[i] >= future_th or (last_buy_time == i - T and instance[i] + prediction[i] >= gamma)):
#                    #buy a bahncard
#                    cost += C + beta * instance[i]
#                    last_buy_time = i
#                    solution.append(i)
#                else:
#                    cost += instance[i]
#                    T_recent_regular_cost += instance[i]
#                    regular_cost[i] = instance[i]
#
#            T_recent_cost += instance[i]
#                
#        return (cost, solution)
#        
#
#    if Lambda == 0.4:
#        regular_cost = [0] * len(instance)
#        gamma = C / (1 - beta)
#        cost = 0
#        solution = []
#        T_recent_regular_cost = 0
#        P_recent_regular_cost = 0
#        current_N = 0
#        T_recent_cost = 0
#        P_recent_cost = 0
#        future_th = (math.sqrt(5.0) + 1) / 2 * gamma
#        K = 2.0 * (math.sqrt(5.0) - 1 - 2 * beta) * (1 - beta) / (math.sqrt(5.0) - 1)
#        deno = math.sqrt(beta * beta - 2.0 * beta + 5) + 1 - 3 * beta
#        golden_gamma = (math.sqrt(5.0) - 1) / 2 * gamma
#        cost = 0
#        last_buy_time = -2 * T
#
#        for i in range(0, len(instance) - T):
#            if (i - T >= 0):
#                T_recent_regular_cost -= regular_cost[i - T]
#                T_recent_cost -= instance[i - T]
#
#            if (last_buy_time + T - 1 >= i):
#                cost += beta * instance[i]
#            else:
#                #cal pdp
#                pdp[i] = dp[i]
#                pN[i] = N[i]
#                for j in range(i + 1, i + T):
#                    pdp[j] = math.inf
#                    pN[j] = 0
#
#                    if (j - T < 0):
#                        if C + beta * (pre_sum[i] + npre_sum[j] - npre_sum[i]) < pdp[j]:
#                            pdp[j] = C + beta * (pre_sum[i] + npre_sum[j] - npre_sum[i])
#                            pN[j] = 1
#                    else:
#                        if dp[j - T] + C + beta * (pre_sum[i] - pre_sum[j - T] + npre_sum[j] - npre_sum[i]) < pdp[j]:
#                            pdp[j] = dp[j - T] + C + beta * (pre_sum[i] - pre_sum[j - T] + npre_sum[j] - npre_sum[i])
#                            pN[j] = pN[j - T] + 1
#
#                    if pdp[j - 1] + predicted_instance[j] < pdp[j]:
#                        pdp[j] = pdp[j - 1] + predicted_instance[j]
#                        pN[j] = pN[j - 1]
#
#                if pN[i + T - 1] > current_N:
#                    P_recent_regular_cost = 0
#                    P_recent_cost = 0
#                    current_N = pN[i + T - 1]
#
#                # Case 1: if buy at time i
#                # 1. adv puts a 0 at i + T
#                cost_alg1 = cost + C + beta * (instance[i] + npre_sum[i + T - 1] - npre_sum[i])
#                cost_dp1 = pdp[i + T - 1]
#                # 2. adv puts a gamma at i + T
#                cost_alg2 = cost + C + beta * (instance[i] + npre_sum[i + T - 1] - npre_sum[i]) + gamma
#                cost_dp2 = min(pdp[i + T - 1] + gamma, dp[i] + C + beta * (npre_sum[i + T - 1] - npre_sum[i] + gamma))
#                cr_buy_at_i = max(cost_alg1 / cost_dp1, cost_alg2 / cost_dp2)
#
#                # Case 2: if buy at time i + 1
#                # 1. adv puts 0s at i + T and i + T + 1
#                cost_alg1 = cost + instance[i] + C + beta * (npre_sum[i + T - 1] - npre_sum[i])
#                cost_dp1 = pdp[i + T - 1]
#                # 2. adv puts a 0 at i + T and a gamma at i + T + 1
#                cost_alg2 = cost + instance[i] + C + beta * (npre_sum[i + T - 1] - npre_sum[i]) + gamma
#                cost_dp2 = min(pdp[i + T - 1] + gamma, pdp[i + 1] + C + beta * (npre_sum[i + T - 1] - npre_sum[i + 1] + gamma))
#                cr_buy_at_ip1 = max(cost_alg1 / cost_dp1, cost_alg2 / cost_dp2)
#
#                # Case 3: do not buy anymore, adv puts a 0 at and after i + T
#                cost_alg1 = cost + instance[i] + npre_sum[i + T - 1] - npre_sum[i]
#                cost_dp1 = pdp[i + T - 1] + 0
#                cr_no_buy_at_i = cost_alg1 / cost_dp1
#
#                #if do not buy at time i+1, adv puts 0s at i + T and i + T + 1
#                cost_alg1 = min(cost + instance[i] + predicted_instance[i + 1] + npre_sum[i + T - 1] - npre_sum[i + 1], cost + instance[i] + predicted_instance[i + 1] + C + beta * (npre_sum[i + T - 1] - npre_sum[i + 1]))
#                cost_dp1 = pdp[i + T - 1] + 0 + 0
#                cr_no_buy_at_ip1 = cost_alg1 / cost_dp1
#
#                buy_card = 0
#                past_regular_cost = 0
#                for j in range(0, T):
#                    if i - j < 0:
#                        break
#                    if j > 0:
#                        if regular_cost[i - j] == 0:
#                            continue
#                        past_regular_cost += regular_cost[i - j]
#                    tmp_A = past_regular_cost + instance[i]
#                    tmp_B = npre_sum[i - j + T - 1] - npre_sum[i] + pre_sum[i]
#                    if i - j > 0:
#                        tmp_B -= pre_sum[i - j - 1]
#                    if (tmp_B - past_regular_cost >= max(gamma, (math.sqrt(5.0) - 1 + 2 * beta) / (2 * beta) * gamma - 1.0 * tmp_A / beta)):
#                        buy_card = 1
#                        #print("buy card at {}, j = {}, A = {} gamma, B = {} gamma".format(i, i - j, past_regular_cost / gamma, (instance[i - j] + prediction[i - j] - past_regular_cost) / gamma))
#                        break
#                #if (instance[i] >= ((math.sqrt(5.0) - 1) * C - K * T_recent_regular_cost) / (2.0 * beta * (1 - beta)) and instance[i] + prediction[i] >= gamma):
#                #if (instance[i] >= (C - T_recent_regular_cost * (1.0 - beta)) / (2.0 * (1 - beta)) and instance[i] + prediction[i] >= gamma):
#                #if ((T_recent_regular_cost >= 2 * gamma) or ((T_recent_cost + instance[i]) >= golden_gamma and instance[i] + prediction[i] >= gamma)):
#                if ((P_recent_cost + instance[i]) >= (math.sqrt(2) - 1) * gamma and instance[i] + prediction[i] >= gamma):
#                #if (cr_buy_at_i < cr_buy_at_ip1 and instance[i] + npre_sum[i + T - 1] - npre_sum[i] >= gamma):
#                #if (cr_buy_at_i < cr_buy_at_ip1):
#                    #print("i = {}, buy, instance = {}, future_cost = {}, cr_buy = {}, cr_no_buy = {}".format(i, instance[i], npre_sum[i + T - 1] - npre_sum[i]))
#                    #if (buy_card == 1):
#                    #buy a bahncard
#                    cost += C + beta * instance[i]
#                    last_buy_time = i
#                    solution.append(i)
#                else:
#                    cost += instance[i]
#                    T_recent_regular_cost += instance[i]
#                    P_recent_regular_cost += instance[i]
#                    regular_cost[i] = instance[i]
#            
#            T_recent_cost += instance[i]
#            P_recent_cost += instance[i]
#
#        return (cost, solution)
#
#
#    #algo
#    buys = []
#    cost = 0
#    last_buy_time = -T
#    gamma = C / (1 - beta)
#    past_threshold = (math.sqrt(5.0) - 1) / 2 * gamma
#    future_threshold = (math.sqrt(1 + 4 * beta) - 1 + 2 * beta) / (2 * beta) * gamma
#    target = (math.sqrt(5.0) + 1) / (2 + beta * (math.sqrt(5.0) - 1))
#    #target = (2 * beta + math.sqrt(1 + 4 * beta) - 1) / (beta + beta * math.sqrt(1 + 4 * beta))
#    T_recent_regular_cost = 0
#    regular_cost = [0] * len(instance)
#    bad_flag = -1
#
#    for i in range(0, len(instance) - T):
#        if (i - T >= 0):
#            T_recent_regular_cost -= regular_cost[i - T]
#
#        check = 1
#        if (last_buy_time + T - 1 >= i):
#            cost += beta * instance[i]
#            check = 0
#        else:
#            #cal pdp
#            pdp[i] = dp[i]
#            for j in range(i + 1, i + T):
#                pdp[j] = math.inf
#                if (j - T < 0):
#                    pdp[j] = min(pdp[j], C + beta * (pre_sum[i] + npre_sum[j] - npre_sum[i]))
#                else:
#                    pdp[j] = min(pdp[j], dp[j - T] + C + beta * (pre_sum[i] - pre_sum[j - T] + npre_sum[j] - npre_sum[i]))
#
#                pdp[j] = min(pdp[j], pdp[j - 1] + predicted_instance[j])
#
#            # Case 1: if buy at time i
#            # 1. adv puts a 0 at i + T
#            cost_alg1 = cost + C + beta * (instance[i] + npre_sum[i + T - 1] - npre_sum[i])
#            cost_dp1 = pdp[i + T - 1]
#            # 2. adv puts a gamma at i + T
#            cost_alg2 = cost + C + beta * (instance[i] + npre_sum[i + T - 1] - npre_sum[i]) + gamma
#            cost_dp2 = min(pdp[i + T - 1] + gamma, dp[i] + C + beta * (npre_sum[i + T - 1] - npre_sum[i] + gamma))
#            cr_buy_at_i = max(cost_alg1 / cost_dp1, cost_alg2 / cost_dp2)
#
#            # Case 2: if buy at time i + 1
#            # 1. adv puts 0s at i + T and i + T + 1
#            cost_alg1 = cost + instance[i] + C + beta * (npre_sum[i + T - 1] - npre_sum[i])
#            cost_dp1 = pdp[i + T - 1]
#            # 2. adv puts a 0 at i + T and a gamma at i + T + 1
#            cost_alg2 = cost + instance[i] + C + beta * (npre_sum[i + T - 1] - npre_sum[i]) + gamma
#            cost_dp2 = min(pdp[i + T - 1] + gamma, pdp[i + 1] + C + beta * (npre_sum[i + T - 1] - npre_sum[i + 1] + gamma))
#            cr_buy_at_ip1 = max(cost_alg1 / cost_dp1, cost_alg2 / cost_dp2)
#
#            # Case 3: do not buy anymore, adv puts a 0 at and after i + T
#            cost_alg1 = cost + instance[i] + npre_sum[i + T - 1] - npre_sum[i]
#            cost_dp1 = pdp[i + T - 1]
#            cr_no_buy_at_i = cost_alg1 / cost_dp1
#
#            #buy_card = 0
#            #past_regular_cost = 0
#            #for j in range(0, T):
#            #    if i - j < 0:
#            #        break
#            #    if j > 0:
#            #        if regular_cost[i - j] == 0:
#            #            continue
#            #        past_regular_cost += regular_cost[i - j]
#            #    tmp_A = past_regular_cost + instance[i]
#            #    tmp_B = npre_sum[i - j + T - 1] - npre_sum[i] + pre_sum[i]
#            #    if i - j > 0:
#            #        tmp_B -= pre_sum[i - j - 1]
#            #    if (tmp_B - past_regular_cost >= max(gamma, (math.sqrt(5.0) - 1 + 2 * beta) / (2 * beta) * gamma - 1.0 * tmp_A / beta)):
#            #        buy_card = 1
#            #        #print("buy card at {}, j = {}, A = {} gamma, B = {} gamma".format(i, i - j, past_regular_cost / gamma, (instance[i - j] + prediction[i - j] - past_regular_cost) / gamma))
#            #        break
#
#            T_past_cost = pre_sum[i]
#            if i - T >= 0:
#                T_past_cost -= pre_sum[i - T]
#
#            #if (cr_2 > target and buy_card == 1):
#            #if cr_2 > arget and cr_1 <= target:
#            #if (cr_1 <= target and instance[i] >= (C - T_recent_regular_cost * (1.0 - beta)) / (2.0 * (1 - beta)) and instance[i] + prediction[i] >= gamma):
#            #if (cr_2 > target or (cr_1 <= target and instance[i] >= (C - T_recent_regular_cost * (1.0 - beta)) / (2.0 * (1 - beta)) and instance[i] + prediction[i] >= gamma)):
#            #if (cr_2 > target or (cr_1 <= target and instance[i] + prediction[i] >= gamma)):
#            #if cr_1 <= target and instance[i] + c_ip1_to_ipTm1 >= gamma and T_past_cost >= gamma:
#            #if (cr_buy_at_i <= target and cr_buy_at_ip1 >= target and instance[i] + npre_sum[i + T - 1] - npre_sum[i] >= gamma):
#            if (cr_buy_at_i < cr_buy_at_ip1 and instance[i] + npre_sum[i + T - 1] - npre_sum[i] >= gamma):
#                #if (cr_buy_at_i > target):
#                    #print("bigger than target {}, val = {}".format(target, cr_buy_at_i))
#                    #bad_flag = i
#                #buy
#                #if T_past_cost < gamma or instance[i] + c_ip1_to_ipTm1 < gamma:
#                    #print("buy, ft = {}, beta = {}, C = {}, gamma = {}, T-past-cost = {}, ins = {}, T-future-cost = {}".format(future_threshold, beta, C, round(gamma, 2), T_past_cost, instance[i], instance[i] + c_ip1_to_ipTm1))
#                    #print("i = {}, best_c1 = {}, best_dp1 = {}, ratio = {}, target = {}".format(i, best_cost1, best_dp1, best_cost1 * 1.0 / best_dp1, target))
#                    #print("i = {}, best_c2 = {}, best_dp2 = {}, ratio = {}, target = {}".format(i, best_cost2, pdp[i + T - 1], best_cost2 * 1.0 / pdp[i + T - 1], target))
#                cost += C + beta * instance[i]
#                last_buy_time = i
#                buys.append(i)
#                check = 0
#            else:
#                cost += instance[i]
#                T_recent_regular_cost += instance[i]
#                regular_cost[i] = instance[i]
#
#                #buy
#                #if (instance[i] + npre_sum[min(i + T - 1, len(instance) - 1)] - npre_sum[i] >= future_threshold):
#                #    pre_cal_cost = cost + C + beta * (instance[i] + npre_sum[min(i + T - 1, len(instance) - 1)] - npre_sum[i])
##print("i = {}, #pre_cal_cost = {}, dp = {}".format(i, pre_cal_cost, min(dp[min(i + T - 1, len(instance) - 1)][0], dp[min(i + T - 1, len(instance) - 1)][1])))
#                #    if (pre_cal_cost * 1.0 / pdp[min(i + T - 1, len(instance) - 1)] > target):
#                #        # do not buy
#                #        cost += instance[i]
#                #    else:
#                #        past_T_cost = pre_sum[i]
#                #        if (i - T >= 0):
#                #            past_T_cost -= pre_sum[i - T]
#                #        if past_T_cost >= 0:
#                #            cost += C + beta * instance[i]
#                #            last_buy_time = i
#                #            buys.append(i)
#                #            check = 0
#                #        else:
#                #            cost += instance[i]
#                #else:
#                #    cost += instance[i]
#
#        #if (check == 1 and cost > 0 and cost * 1.0 / dp[i] > target):
#         #   print("Exceeded, Lambda = {}, T = {}, C = {}, beta = {}, i = {}, cost = {}, dp = {}, ratio = {}, target = {}".format(Lambda, T, C, beta, i, cost, dp[i], cost * 1.0 / dp[i], target))
#
#    if bad_flag >= 0:
#        print("bad_pos = {}, instance = {}, solution = {}, cost = {}, opt = {}".format(bad_flag, instance, buys, cost, dp[len(instance) - T - 1]))
#    return (cost, buys)


#PFSUM purchases a bahncard at time t whenever (i) T-recent-cost is at least gamma, and (ii) the predicted T-future-cost is also at least gamma.
def PFSUM(instance, T, C, beta, prediction):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    cost = 0
    solution = []
    last_buy_time = -T
    T_recent_cost = 0

    for i in range(0, length):
        if (i - T >= 0):
            T_recent_cost -= instance[i - T]

        if (last_buy_time + T - 1 >= i):
            cost += beta * instance[i]
        else:
			#prediction[i] means the predicted cost in (i, i + T)
            if (T_recent_cost + instance[i] >= gamma and instance[i] + prediction[i] >= gamma):
                #buy a bahncard
                cost += C + beta * instance[i]
                last_buy_time = i
                solution.append(i)
            else:
                cost += instance[i]

        T_recent_cost += instance[i]
            
    return (cost, solution)

#Online primal-dual learning augmented algorithm for the bahncard problem, which follows Algorithm 8 of Bamas's paper published at NeurIPS 2020.
def PDLA_FOR_BAHNCARD(instance, T, C, beta, Lambda, predicted_solution):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    c_lambda = (1 + 1 / gamma)**(Lambda*gamma)
    c_1_by_lambda = (1 + 1 / gamma)**(gamma/Lambda)

    pre_x_sum = [0] * length
    d = []
    f = []
    cost = 0
    solution = []
    latest_predicted_idx = -1
    last_buy_time = -T
    
    for i in range(0, length):
        while (latest_predicted_idx + 1 < len(predicted_solution)
            and predicted_solution[latest_predicted_idx + 1] <= i):
            latest_predicted_idx += 1

        if (i > 0):
            pre_x_sum[i] = pre_x_sum[i - 1]

        T_recent_x_sum = pre_x_sum[i]
        if (i - T >= 0):
            T_recent_x_sum -= pre_x_sum[i - T]

        if (instance[i] == 0):
            continue

        #a request arrived at time i
        if (T_recent_x_sum >= 1):
            if (last_buy_time + T - 1 < i):
                cost += C
                last_buy_time = i

            #for a minimal update, the primal cost is beta * price
            cost += beta * instance[i]
        else:
            if (latest_predicted_idx >= 0 and predicted_solution[latest_predicted_idx] + T - 1 >= i):
                #big update
                x_increment = instance[i] * (T_recent_x_sum + 1 / (c_lambda - 1)) / gamma
                pre_x_sum[i] += x_increment

                #primal cost brought by the increase in x
                cost += instance[i]
            else:
                #small update
                x_increment = instance[i] * (T_recent_x_sum + 1 / (c_1_by_lambda - 1)) / gamma
                pre_x_sum[i] += x_increment

                #primal cost brought by the increase in x
                cost += instance[i]

            solution.append((i, x_increment))

    return (cost, solution)

#PDLA based on fractional solutions. It will not be included in the experiments.
def Fractional_PDLA_FOR_BAHNCARD(instance, T, C, beta, Lambda, predicted_solution):
    length = len(instance)
    if (length == 0):
        return (0, [])

    gamma = C / (1 - beta)
    c_lambda = (1 + 1 / gamma)**(Lambda*gamma)
    c_1_by_lambda = (1 + 1 / gamma)**(gamma/Lambda)

    pre_x_sum = [0] * length
    d = []
    f = []
    cost = 0
    solution = []
    latest_predicted_idx = -1
    
    for i in range(0, length):
        while (latest_predicted_idx + 1 < len(predicted_solution)
            and predicted_solution[latest_predicted_idx + 1] <= i):
            latest_predicted_idx += 1

        if (i > 0):
            pre_x_sum[i] = pre_x_sum[i - 1]

        T_recent_x_sum = pre_x_sum[i]
        if (i - T >= 0):
            T_recent_x_sum -= pre_x_sum[i - T]

        if (instance[i] == 0):
            continue

        #a request arrived at time i
        if (T_recent_x_sum >= 1):
            #for a minimal update, the primal cost is beta * price
            cost += beta * instance[i]
        else:
            if (latest_predicted_idx >= 0 and predicted_solution[latest_predicted_idx] + T - 1 >= i):
                #big update
                x_increment = instance[i] * (T_recent_x_sum + 1 / (c_lambda - 1)) / gamma
                pre_x_sum[i] += x_increment

                #primal cost brought by the increase in x
                cost += instance[i] * (1 - beta) / (c_lambda - 1) + instance[i]
            else:
                #small update
                x_increment = instance[i] * (T_recent_x_sum + 1 / (c_1_by_lambda - 1)) / gamma
                pre_x_sum[i] += x_increment

                #primal cost brought by the increase in x
                cost += instance[i] * (1 - beta) / (c_1_by_lambda - 1) + instance[i]

            solution.append((i, x_increment))

    return (cost, solution)


# Offline optimal algorithm return the optimal cost and the corresponding time list of buying bahncard. The algorithm is based on dynamic programming.
def OFFLINE_OPTIMAL(instance, T, C, beta):
    length = len(instance)
    if (length == 0):
        return (0, [])

    #dp[i][1] indicates the optimal cost for all requests arrived in time interval [0, i], and a bahncard expires at time i. Besides, there is no bahncard expire at time i in the case of dp[i][0].
    dp = []
    #dp_pre[i][0/1] records the suboptimal structure of dp[i][0/1]
    dp_pre = []

    pre_sum = [instance[0]]
    for i in range(1, length):
        pre_sum.append(pre_sum[i - 1] + instance[i])

    dp.append([instance[0], C + beta * instance[0]])
    dp_pre.append([[-1, -1], [-1, -1]])

    for i in range(1, length):
        dp.append([math.inf, math.inf])
        dp_pre.append([[-1, -1], [-1, -1]])

        if (i - T < 0):
            dp[i][1] = C + beta * pre_sum[i]
        else:
            if (dp[i - T][0] < dp[i - T][1]):
                dp[i][1] = dp[i - T][0] + C + beta * (pre_sum[i] - pre_sum[i - T])
                dp_pre[i][1] = [i - T, 0]
            else:
                dp[i][1] = dp[i - T][1] + C + beta * (pre_sum[i] - pre_sum[i - T])
                dp_pre[i][1] = [i - T, 1]

        if (dp[i - 1][0] < dp[i - 1][1]):
            dp[i][0] = dp[i - 1][0] + instance[i]
            dp_pre[i][0] = [i - 1, 0]
        else:
            dp[i][0] = dp[i - 1][1] + instance[i]
            dp_pre[i][0] = [i - 1, 1]

    p = length - 1
    idx = 0
    optimal_cost = dp[length - 1][0]
    if (dp[length - 1][1] < dp[length - 1][0]):
        idx = 1
        optimal_cost = dp[length - 1][1]

    optimal_solution = []

    #restore the optimal solution
    while p != -1:
        if (idx == 1):
            optimal_solution.append(max(0, p - T + 1))

        new_p = dp_pre[p][idx][0]
        new_idx = dp_pre[p][idx][1]
        p = new_p
        idx = new_idx

    optimal_solution.reverse()

    return (optimal_cost, optimal_solution)

