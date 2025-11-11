import numpy as np
from problems.lockwood import runlock, runlock_splitted
import pycutest
import time


def LOCKWOOD1(n=6, pause=0):

    f_list = []
    I = []
   
    if pause == 0:
        f_list.append(lambda x: runlock_splitted(x, 0)[0])
        f_list.append(lambda x: runlock_splitted(x, 1)[1])
        f_list.append(lambda x: runlock_splitted(x, 2)[2])
    else:
        f_list.append(lambda x: (time.sleep(pause),  runlock_splitted(x, 0)[0])[1])
        f_list.append(lambda x: (time.sleep(pause),  runlock_splitted(x, 1)[1])[1])
        f_list.append(lambda x: (time.sleep(pause),  runlock_splitted(x, 2)[2])[1])

    I.append([i for i in range(n)])
    I.append([i for i in range(n)])
    I.append([i for i in range(n)])

    x0 = np.repeat(10000, n)
    ub = np.repeat(10000, n)
    lb = np.zeros(n)

    return f_list, x0, I, lb, ub


def LOCKWOOD2(n, pause=0):

    assert n % 5 == 1

    rng = np.random.default_rng(42)

    f_list = []
    I = []

    for i in range(n // 5):

        l_vals = rng.random(3)
        l_vals = l_vals / sum(l_vals)

        if pause == 0:
            f_list.append(lambda x, i=i: runlock(x[i * 5 : (i + 1) * 5 + 1], l_vals, i))
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), runlock(x[i * 5 : (i + 1) * 5 + 1], l_vals, i))[1])
    
        I.append([i*5, (i*5)+1, (i*5)+2, (i*5)+3, (i*5)+4, (i*5)+5])

    x0 = np.repeat(10000, n)
    ub = np.repeat(10000, n)
    lb = np.zeros(n)

    return f_list, x0, I, lb, ub


def CUTEST(prob_name, n, constrained=False, pause=0): 
    """ 
    Makes overlapped replicae of the original cutest problems. 
    The last variable of a subfunction is used as the first.
    """

    prob_properties = pycutest.problem_properties(prob_name)

    prob = pycutest.import_problem(prob_name)
    n_cutest = prob.n

    assert (n - n_cutest) % (n_cutest - 1) == 0 and prob.x0[0] == prob.x0[-1]

    if constrained:
        assert prob_properties['constraints'] == 'bound' and prob.bl[0] == prob.bl[-1] and prob.bu[0] == prob.bu[-1]

    else:
        assert prob_properties['constraints'] == 'unconstrained'

    f_list = []
    I = []

    for i in range(int((n - n_cutest) / (n_cutest - 1)) + 1):

        if pause == 0:
            f_list.append(lambda x, i=i: prob.obj(x[i*(n_cutest - 1) : (i+1) * (n_cutest - 1) + 1]))
        
        else: 

            f_list.append(lambda x, i=i:  (time.sleep(pause), prob.obj(x[i*(n_cutest - 1) : (i+1) * (n_cutest - 1) + 1]))[1])

        I.append([i*(n_cutest - 1) + j for j in range(n_cutest)])

    x0 = np.concat([prob.x0] + [prob.x0[1:] for _ in range(int((n - n_cutest) / (n_cutest - 1)))])

    if constrained:
        lb = np.concat([prob.bl] + [prob.bl[1:] for _ in range(int((n - n_cutest) / (n_cutest - 1)))])
        ub = np.concat([prob.bu] + [prob.bu[1:] for _ in range(int((n - n_cutest) / (n_cutest - 1)))])

        for i in range(len(x0)):
            x0[i] = max(min(x0[i], ub[i]), lb[i])

    else:
        lb, ub = None, None

    return f_list, x0, I, lb, ub


def CUTEST_COMPOSITE(list_of_prob_names, n_shared, constrained=False, pause=0):
    """ 
    Makes partially separable problems with objective the sum of m cutest problems. 
    The first n_shared variables are common to all subproblems.
    """

    cutest_problems = {}
    total_vars = 0
    
    for prob_name in list_of_prob_names:

        prob_properties = pycutest.problem_properties(prob_name)

        if constrained:
            assert prob_properties['constraints'] == 'bound'

        else:
            assert prob_properties['constraints'] == 'unconstrained'

        cutest_problems[prob_name] = pycutest.import_problem(prob_name)

        total_vars += cutest_problems[prob_name].n
    
    total_vars = total_vars - n_shared * (len(list_of_prob_names) - 1)

    x0 = np.zeros(total_vars)

    if constrained:
        lb = np.full_like(x0, np.nan)
        ub = np.full_like(x0, np.nan)
    else:
        lb, ub = None, None

    f_list = []
    I = []
    counter = n_shared

    for prob_name, prob in cutest_problems.items():

        if pause == 0:
            f_list.append(lambda x, counter=counter: prob.obj(np.concat((x[ : n_shared], x[counter : counter + prob.n - n_shared])))  )
        
        else: 
            f_list.append(lambda x, counter=counter: (time.sleep(pause), prob.obj(np.concat((x[ : n_shared], x[counter : counter + prob.n - n_shared]))))[1])

        I.append([j for j in range(n_shared)] + [j for j in range(counter, counter + prob.n - n_shared)])

        x0[: n_shared] = x0[: n_shared] + prob.x0.copy()[:n_shared]
        x0[counter : counter + prob.n - n_shared] = prob.x0.copy()[n_shared:]

        if constrained:
            
            if np.isnan(lb[:n_shared]).all():
                lb[:n_shared] = prob.bl.copy()[:n_shared]
            else:
                lb[:n_shared] = np.maximum(prob.bl.copy()[:n_shared], lb[:n_shared])

            if np.isnan(ub[:n_shared]).all():
                ub[:n_shared] = prob.bu.copy()[:n_shared]

            else:
                ub[:n_shared] = np.minimum(prob.bu.copy()[:n_shared], ub[:n_shared])

            lb[counter : counter + prob.n - n_shared] = prob.bl.copy()[n_shared:]
            ub[counter : counter + prob.n - n_shared] = prob.bu.copy()[n_shared:]

        counter +=  prob.n - n_shared

    x0[: n_shared] = x0[: n_shared] / len(f_list)

    if constrained:
        if (lb[:n_shared] > ub[:n_shared]).any():
            raise ValueError("The feasible set is empty!")

        for i in range(len(x0)):
            x0[i] = max(min(x0[i], ub[i]), lb[i])

    return f_list, x0, I, lb, ub
