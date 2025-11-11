import numpy as np
import os
import time
from log_utils import save_json, init_csv, append_row_csv

def line_search(f_tot, x0, lb, ub, n_functions,
                toll, delta, gamma, maxit, max_eval, pause, time_limit, out_dir,
                for_hybrid=False, hyb_k=0, hyb_time=0, hyb_eval_count=0, hyb_subf_count=0, hyb_alpha=1):

    save_json(os.path.join(out_dir, 'params.json'), 
               {'toll': toll,
                'maxit': maxit,
                'delta': delta,
                'gamma': gamma,
                'pause': pause,
                'time_limit': time_limit,
                'max_eval': max_eval})

    if lb is None:
        lb = -np.inf * np.ones(len(x0))
    if ub is None:
        ub = np.inf * np.ones(len(x0))

    full_time = 0
    s_time = time.time()

    n = len(x0)
    d = np.eye(n)
    
    if for_hybrid:
        alpha = np.full(n, hyb_alpha)
    else:
        alpha = np.ones(n)
    k = 0
    
    x = x0.copy()
    func_eval_count = 0
    
    current_f = f_tot(x)
    func_eval_count += 1
    
    x_next = None

    e_time = time.time()

    full_time += e_time - s_time

    # LOGS are not considered on time
    if not for_hybrid:
        col_names = ['k', 'f', 'time', 'norm_update', 'max_alpha', 'count', 'sum_subf_count'] 
        init_csv(os.path.join(out_dir), 'log.csv', col_names)

    print('k     f      time    norm_update   max_alpha   count    sum_subf_count' )
    print(f"{k}   {current_f:.6f} {full_time:.6f}  {0:.6f}  {np.max(alpha):.6f}      {func_eval_count}      {func_eval_count * n_functions}")

    if not for_hybrid:
        out_row = [k, current_f, full_time, 0, np.max(alpha), func_eval_count, func_eval_count * n_functions]
        append_row_csv(os.path.join(out_dir), 'log.csv', out_row)
    
    while k < maxit and full_time < time_limit:
        s_time = time.time()

        prev_x = x.copy()
        
        for i in range(n):
            amax_pos = ub[i] - x[i]
            a_pos = min(alpha[i], amax_pos)
            
            amax_neg = x[i] - lb[i]
            a_neg = min(alpha[i], amax_neg)
            
            if a_pos > 0:

                x_tent = x + a_pos * d[:, i] # positive direction
                f_tent = f_tot(x_tent)
                func_eval_count += 1

                if f_tent == current_f:
                    pass
                
                if f_tent <= current_f - gamma * (a_pos ** 2) and f_tent != current_f and not np.isnan(f_tent): # suff. decrement
                    a = a_pos
                    x_next = x_tent.copy()
                    f_next = f_tent
                    
                    while True:
                        x_tent = x + (a / delta) * d[:, i]
                        f_tent = f_tot(x_tent)
                        func_eval_count += 1
                        
                        if f_tent > current_f - gamma * ((a / delta) ** 2) or (a / delta > amax_pos) or f_tent == current_f or np.isnan(f_tent):
                            break
                        
                        a /= delta
                        x_next = x_tent.copy()
                        f_next = f_tent
                    
                    alpha[i] = a
                    x = x_next.copy()
                    current_f = f_next
                else:
                    x_next = None
            
            if a_neg > 0 and not np.array_equal(x, x_next):

                x_tent = x - a_neg * d[:, i] # negative direction
                f_tent = f_tot(x_tent)
                func_eval_count += 1

                if f_tent == current_f:
                    pass
                
                if f_tent <= current_f - gamma * (a_neg ** 2) and f_tent != current_f and not np.isnan(f_tent):
                    a = a_neg
                    x_next = x_tent.copy()
                    f_next = f_tent
                    
                    while True:
                        x_tent = x - (a / delta) * d[:, i]
                        f_tent = f_tot(x_tent)
                        func_eval_count += 1
                        
                        if f_tent > current_f - gamma * ((a / delta) ** 2) or (a / delta > amax_neg) or f_tent == current_f or np.isnan(f_tent):
                            break
                        
                        a /= delta
                        x_next = x_tent.copy()
                        f_next = f_tent
                    
                    alpha[i] = a
                    x = x_next.copy()
                    current_f = f_next
                else:
                    x_next = None
            
            if (a_neg <= 0 and a_pos <= 0) or x_next is None:
                alpha[i] *= delta

        termination = False
        if (np.max(alpha) <= toll and np.linalg.norm(x - prev_x) < toll) or func_eval_count >= max_eval:
            termination = True

        k += 1

        e_time = time.time()
        full_time += e_time - s_time

        if k % 100 == 0 or termination:
            print(f"{k}   {current_f:.6f} {full_time:.6f}  {np.linalg.norm(x - prev_x):.6f}  {np.max(alpha):.6f}    {func_eval_count}        {func_eval_count * n_functions}")

        if for_hybrid:
            out_row = [k + hyb_k, current_f, full_time + hyb_time, np.linalg.norm(x - prev_x), np.max(alpha), 0, func_eval_count + hyb_eval_count, func_eval_count * n_functions + hyb_subf_count, 0]
        else: 
            out_row = [k, current_f, full_time, np.linalg.norm(x - prev_x), np.max(alpha), func_eval_count, func_eval_count * n_functions] #  + [a for a in alpha]
        
        append_row_csv(os.path.join(out_dir), 'log.csv', out_row)

        if termination:
            break
    
    if k == maxit:
        print("Reached MAX Iter!")
    if full_time >= time_limit:
        print("Reached Time Limit")
    
    return x, {'k': k, 'f': current_f, 't': full_time, 'max_alpha': np.max(alpha), 'f_eval': func_eval_count, 'f_eval_sub': func_eval_count * n_functions}