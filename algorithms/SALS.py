import numpy as np
import os
import time
from log_utils import save_json, init_csv, append_row_csv

def structure_aware_LS(f_list, I, x0, lb, ub, toll, delta, gamma, maxit, max_subf_eval, pause, time_limit, out_dir,
                                for_hybrid=False, hyb_k=0, hyb_time=0, hyb_eval_count=0, hyb_subf_count=0, hyb_alpha=1):
    
    save_json(os.path.join(out_dir, 'params.json'), 
               {'toll': toll,
                'maxit': maxit,
                'delta': delta,
                'gamma': gamma,
                'pause': pause,
                'time_limit': time_limit,
                'max_subf_eval': max_subf_eval})
    
    if lb is None:
        lb = -np.inf * np.ones(len(x0))
    if ub is None:
        ub = np.inf * np.ones(len(x0))

    full_time = 0
    s_time = time.time()

    n = len(x0)
    q = len(f_list)
    d = np.eye(n)
    
    if for_hybrid:
        alpha = np.full(n, hyb_alpha)
    else:
        alpha = np.ones(n)
    k = 0

    x = x0.copy()
    count = np.zeros(q)
    
    current_f_vals = np.array([f(x) for f in f_list])
    count += 1 

    e_time = time.time()
    full_time += e_time - s_time

    # LOGS are not considered on time
    if not for_hybrid:
        col_names = ['k', 'f', 'time', 'norm_update', 'max_alpha', 'min_count', 'avg_count', 'sum_count'] # + ['alpha' + str(i) for i in range(n)] + ['f' + str(i) for i in range(q)] + ['count' + str(i) for i in range(q)]
        init_csv(os.path.join(out_dir), 'log.csv', col_names)

    print('k         f           time      norm_update  max_alpha  min_count  avg_count   sum_count' )
    print(f"{k}  {sum(current_f_vals):.6f}   {full_time:.6f}  {0:.6f}  {np.max(alpha):.6f} {np.min(count)}   {np.mean(count)}   {np.mean(count)}")
    
    if not for_hybrid:
        out_row = [k, sum(current_f_vals), full_time, 0, np.max(alpha), np.min(count), np.mean(count)] # + [a for a in alpha] + [f_i_val for f_i_val in current_f_vals] + [c for c in count]
        append_row_csv(os.path.join(out_dir), 'log.csv', out_row)
    
    while k < maxit and full_time < time_limit:
        s_time = time.time()

        prev_x = x.copy()
        
        for j in range(n):
            f_idxs_depend_on_j = [i for i, indices in enumerate(I) if j in indices]
            current_f = np.sum(current_f_vals)
            
            amax_pos = ub[j] - x[j]
            a_pos = min(alpha[j], amax_pos)
            
            amax_neg = x[j] - lb[j]
            a_neg = min(alpha[j], amax_neg)
            
            x_next = None
            
            # Positive direction
            if a_pos > 0:
                x_tent = x + a_pos * d[:, j]
                f_vals_tent, count = eval_tentative_f_val(x_tent, current_f_vals, f_list, f_idxs_depend_on_j, count)
                
                if np.sum(f_vals_tent) <= current_f - gamma * a_pos**2 and np.sum(f_vals_tent) != current_f and not np.isnan(f_vals_tent).any():
                    a = a_pos
                    x_next, f_vals_next = x_tent, f_vals_tent
                    
                    while True:
                        x_tent = x + (a / delta) * d[:, j]
                        f_vals_tent, count = eval_tentative_f_val(x_tent, current_f_vals, f_list, f_idxs_depend_on_j, count)
                        
                        if np.sum(f_vals_tent) > current_f - gamma * (a / delta) ** 2 or (a / delta) > amax_pos or np.sum(f_vals_tent) == current_f or np.isnan(f_vals_tent).any():
                            break
                        
                        a /= delta
                        x_next, f_vals_next = x_tent, f_vals_tent
                    
                    alpha[j] = a
                    x, current_f_vals = x_next, f_vals_next
            
            # Negative direction
            if a_neg > 0 and not np.array_equal(x, x_next):
                x_tent = x - a_neg * d[:, j]
                f_vals_tent, count = eval_tentative_f_val(x_tent, current_f_vals, f_list, f_idxs_depend_on_j, count)
                
                if np.sum(f_vals_tent) <= current_f - gamma * a_neg**2 and np.sum(f_vals_tent) != current_f and not np.isnan(f_vals_tent).any():
                    a = a_neg
                    x_next, f_vals_next = x_tent, f_vals_tent
                    
                    while True:
                        x_tent = x - (a / delta) * d[:, j]
                        f_vals_tent, count = eval_tentative_f_val(x_tent, current_f_vals, f_list, f_idxs_depend_on_j, count)
                        
                        if np.sum(f_vals_tent) > current_f - gamma * (a / delta) ** 2 or (a / delta) > amax_neg or np.sum(f_vals_tent) == current_f or np.isnan(f_vals_tent).any():
                            break
                        
                        a /= delta
                        x_next, f_vals_next = x_tent, f_vals_tent
                    
                    alpha[j] = a
                    x, current_f_vals = x_next, f_vals_next
            
            if (a_neg == 0 and a_pos == 0) or x_next is None:
                alpha[j] *= delta
        

        termination = False
        if (np.max(alpha) <= toll and np.linalg.norm(x - prev_x) < toll) or np.sum(count) >= max_subf_eval:
            termination = True
        
        k += 1

        e_time = time.time()
        full_time += e_time - s_time

        # LOGS are not considered on time
        if k % 100 == 0 or termination:
            print(f"{k}  {sum(current_f_vals):.6f}   {full_time:.6f}  {np.linalg.norm(x - prev_x):.6f}  {np.max(alpha):.6f} {int(np.min(count))} {np.mean(count):.4f}    {int(np.mean(count))}")
        
        if for_hybrid:
            out_row = [k + hyb_k, sum(current_f_vals), full_time + hyb_time, np.linalg.norm(x - prev_x),
                       np.max(alpha), 0, np.mean(count) + hyb_eval_count, np.sum(count) + hyb_subf_count, 0] 
        else:
            out_row = [k, sum(current_f_vals), full_time, np.linalg.norm(x - prev_x), np.max(alpha), np.min(count), np.mean(count), np.sum(count)] # + [a for a in alpha] + [f_i_val for f_i_val in current_f_vals] + [c for c in count]
        
        append_row_csv(os.path.join(out_dir), 'log.csv', out_row)

        if termination:
            break
    
    if k == maxit:
        print("Reached MAX Iter!")
    if full_time >= time_limit:
        print("Reached Time Limit")
    
    return x, {'k': k, 'f': np.sum(current_f_vals), 't': full_time, 'max_alpha': np.max(alpha), 'f_eval_sub_mean': np.mean(count), 'f_eval_sub': np.sum(count)}

def eval_tentative_f_val(x_tent, current_f_vals, f_list, f_idxs_depend_on_j, count):
    f_vals_tent = current_f_vals.copy()
    for i in f_idxs_depend_on_j:
        f_vals_tent[i] = f_list[i](x_tent)
        count[i] += 1
    return f_vals_tent, count