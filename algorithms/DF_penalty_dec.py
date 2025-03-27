import numpy as np
import os
import time
from log_utils import save_json, init_csv, append_row_csv

def DF_penalty_dec(f_list, x0, I, tau, max_tau, theta,
                   toll, maxit, pause, time_limit,
                   delta, gamma, f_tot_for_logs, out_dir):
    
    save_json(os.path.join(out_dir, 'params.json'), 
               {
                    'tau0': tau,
                    'max_tau': max_tau,
                    'theta': theta,
                    'toll': toll,
                    'maxit': maxit,
                    'pause': pause,
                    'time_limit': time_limit,
                    'delta': delta,
                    'gamma': gamma,
                })
    
    full_time = 0
    s_time = time.time()

    n = len(x0)
    q = len(f_list)
    d = np.eye(n) 

    # Initialize alpha
    alpha = np.zeros((q, n))
    for i in range(q):
        for j in I[i]:
            alpha[i, j] = 1.0
    
    k = 0
    v = np.tile(x0[:, np.newaxis], q) # replicate x0, q times
    x = x0.copy()

    count = np.zeros(q)
    
    e_time = time.time()
    full_time += e_time - s_time


    # LOGS are not considered on time eval
    col_names = ['k', 'f', 'time', 'norm_update', 'max_alpha', 'min_count', 'avg_count', 'tau']
    init_csv(os.path.join(out_dir), 'log.csv', col_names)

    print('k      f      time    norm_update  max_alpha  min_count  avg_count tau')
    print(f"{k}  {f_tot_for_logs(x):.6f}  {full_time:.6f}       0  {np.max(alpha):.6f}  {np.min(count)}  {np.mean(count)}  {tau:.4f}")
    
    out_row = [k, f_tot_for_logs(x), full_time, 0, np.max(alpha), np.min(count), np.mean(count), tau]
    append_row_csv(os.path.join(out_dir), 'log.csv', out_row)


    while k < maxit and full_time < time_limit:

        s_time = time.time()

        prev_x = x.copy()

        for i in range(q):
            f_i = f_list[i]
            current_v = v[:, i].copy()
            
            def q_fun(a, b):
                return f_i(a) + 0.5 * tau * np.sum((a[I[i]] - b[I[i]]) ** 2)
            
            for j in I[i]:
                
                if alpha[i, j] > 0:
                    current_q_val = q_fun(current_v, x)
                    count[i] += 1

                    v_tent = current_v + alpha[i, j] * d[:, j] # positive direction
                    q_tent = q_fun(v_tent, x)
                    count[i] += 1
                    
                    if q_tent <= current_q_val - gamma * (alpha[i, j] ** 2):
                        next_v, a, n_exp = expansion_step(q_fun, current_q_val, current_v, x, v_tent, d[:, j], alpha[i, j], gamma, delta)
                        alpha[i, j] = a
                        current_v = next_v

                    else:
                        v_tent = current_v - alpha[i, j] * d[:, j] # negative direction
                        q_tent = q_fun(v_tent, x)
                        count[i] += 1
                        
                        if q_tent <= current_q_val - gamma * (alpha[i, j] ** 2):
                            next_v, a, n_exp = expansion_step(q_fun, current_q_val, current_v, x, v_tent, -d[:, j], alpha[i, j], gamma, delta)
                            alpha[i, j] = a
                            current_v = next_v
                        else:
                            n_exp = 0
                            alpha[i, j] *= delta
                else:
                    n_exp = 0
                    alpha[i, j] *= delta
                
                count[i] += n_exp
        
            v[:, i] = current_v
        
        x = exact_minimum(I, v, n)
        
        tau = min(theta * tau, max_tau)

        norm_update = np.linalg.norm(x - prev_x)
        max_alpha = np.max(alpha)

        termination = False

        if norm_update < toll and max_alpha < toll / max(1, tau):
            termination = True
    
        k += 1

        e_time = time.time()
        full_time += e_time - s_time
        
        if k % 10 == 0 or termination:
            print(f"{k}  {f_tot_for_logs(x):.6f}  {full_time:.6f}  {norm_update:.6f}  {max_alpha:.6f}  {np.min(count)}  {np.mean(count)}  {tau:.4f}")
        
        out_row = [k, f_tot_for_logs(x), full_time, norm_update, max_alpha, np.min(count), np.mean(count), tau] 
        append_row_csv(os.path.join(out_dir), 'log.csv', out_row)

        if termination:
            break
    
    if k == maxit:
        print("Reached MAX Iter!")
    if full_time >= time_limit:
        print("Reached Time Limit")
    
    return x, k, np.sum(count)


def expansion_step(q_fun, current_q_val, current_v, x, v_tent, d, step, gamma, delta):
    v_next = v_tent.copy()
    
    v_tent = current_v + (step / delta) * d
    q_tent = q_fun(v_tent, x)
    n_exp = 1
    
    while q_tent <= current_q_val - gamma * (step / delta) ** 2:
        step /= delta
        v_next = v_tent.copy()
        
        v_tent = current_v + (step / delta) * d
        q_tent = q_fun(v_tent, x)
        n_exp += 1
    
    return v_next, step, n_exp


def exact_minimum(I, v, n):
    avg_v_val = np.zeros(n)
    for j in range(n):
        indices = [i for i, I_i in enumerate(I) if j in I_i]
        v_vals = v[j, indices]
        avg_v_val[j] = np.mean(v_vals)
    return avg_v_val