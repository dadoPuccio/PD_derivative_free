import numpy as np
import os
import time
from log_utils import save_json, init_csv, append_row_csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from algorithms.base_DF_penalty_dec import coordinate_search, coordinate_search_constr, exact_minimum, exact_minimum_costr, update_alpha


def process_function(i, v_i, I_i, lb, ub, alpha_i, x, tau, gamma, delta, n, constr_policy):
    current_v = v_i.copy()

    d = np.eye(n)

    def q_fun(a, b):
        return F_LIST[i](a) + 0.5 * tau * np.sum((a[I_i] - b[I_i]) ** 2)

    current_q_val = None

    count_i = 0
    for j in I_i:

        if constr_policy == 'internal':
            current_v, current_q_val, alpha_ij, count_i, n_exp = coordinate_search_constr(current_v, alpha_i[j], q_fun, x, count_i, gamma, delta, d[:, j], lb[j], ub[j], j, current_q_val=current_q_val)
        else:
            current_v, current_q_val, alpha_ij, count_i, n_exp = coordinate_search(current_v, alpha_i[j], q_fun, x, count_i, gamma, delta, d[:, j], current_q_val=current_q_val)

        alpha_i[j] = alpha_ij
        count_i += n_exp

    return i, current_v, alpha_i, count_i


def DF_penalty_dec_parallel(f_list, x0, I, lb, ub, 
                            tau, max_tau, toll, delta, gamma, theta, sigma, maxit, max_subf_eval,
                            pause, time_limit, f_tot_for_logs, out_dir, constr_policy):
    
    save_json(os.path.join(out_dir, 'params.json'), 
               {'toll': toll, 
                'maxit': maxit, 
                'tau': tau, 
                'max_tau': max_tau,
                'delta': delta, 
                'gamma': gamma, 
                'theta': theta, 
                'sigma': sigma,
                'pause': pause,
                'time_limit': time_limit,
                'max_subf_eval': max_subf_eval,
                'constr_policy': constr_policy})

    full_time = 0.0
    s_time = time.time()

    n = len(x0)
    q = len(f_list)

    alpha = np.zeros((q, n))
    for i in range(q):
        for j in I[i]:
            alpha[i, j] = 1.0
    
    k = 0
    v = np.tile(x0[:, np.newaxis], q)
    x = x0.copy()
    last_x = x.copy()
    termination = False

    count = np.zeros(q)

    replicated_lb = [None] * q if lb is None else [lb] * q
    replicated_ub = [None] * q if ub is None else [ub] * q

    replicated_gamma = np.full(q, gamma)
    replicated_delta = np.full(q, delta)

    e_time = time.time()
    full_time += e_time - s_time


    # LOGS are not considered on time eval
    col_names = ['k', 'f', 'time', 'norm_update', 'max_alpha', 'min_count', 'avg_count', 'sum_count', 'tau']
    init_csv(out_dir, 'log.csv', col_names)

    print('k     f     time     norm_update  max_alpha min_count avg_count    sum_count   tau')
    print(f"{k}  {f_tot_for_logs(x):.6f} {full_time:.6f}  {0:.6f}  {np.max(alpha):.6f}    {int(np.min(count))}    {np.mean(count):.4f}   {int(np.sum(count))}   {tau:.4f}")

    out_row = [k, f_tot_for_logs(x), full_time, 0, np.max(alpha), np.min(count), np.mean(count), tau] 
    append_row_csv(out_dir, 'log.csv', out_row)

    prev_x = np.zeros_like(x)

    global F_LIST 
    F_LIST = f_list
 
    with ProcessPoolExecutor(max_workers=12) as executor:

        while k < maxit and full_time < time_limit:

            s_time = time.time()

            prev_x[:] = x   

            task_args = []
            for i_idx in range(q):
                task_args.append(
                    (i_idx,
                     v[:, i_idx].copy(), 
                     I[i_idx],
                     replicated_lb[i_idx],
                     replicated_ub[i_idx],
                     alpha[i_idx].copy(),  
                     x, 
                     tau,
                     replicated_gamma[i_idx],
                     replicated_delta[i_idx],
                     n,
                     constr_policy)
                )
            
            futures = [executor.submit(process_function, *args) for args in task_args]

            for fut in futures:
                i_res, new_v_i, new_alpha_i, count_i = fut.result()
                v[:, i_res] = new_v_i
                alpha[i_res] = np.asarray(new_alpha_i).reshape(-1)
                count[i_res] += int(count_i)


            if constr_policy in ['external', 'internal']:
                x = exact_minimum_costr(I, v, n, lb, ub)
            else:
                x = exact_minimum(I, v, n)

            update_norm = np.linalg.norm(x - prev_x)
            max_alpha = np.max(alpha)
            
            if update_norm < toll and max_alpha < max((toll * 100) / max(1, tau), toll):
                tau = min(theta * tau, max_tau)
                
                if max_alpha < toll and np.linalg.norm(x - last_x) < toll:
                    termination = True
                else:
                    last_x = x.copy()

            if np.sum(count) >= max_subf_eval:
                termination = True

            k += 1

            e_time = time.time()
            full_time += e_time - s_time
            
            if k % 100 == 0 or termination:
                print(f"{k}  {f_tot_for_logs(x):.6f} {full_time:.6f}  {update_norm:.6f}  {np.max(alpha):.6f}    {int(np.min(count))}    {np.mean(count):.4f}    {int(np.sum(count))}   {tau:.4f}")
            
            out_row = [k, f_tot_for_logs(x), full_time, update_norm, max_alpha, np.min(count), np.mean(count), np.sum(count), tau]
            append_row_csv(os.path.join(out_dir), 'log.csv', out_row)

            if termination:
                break

    if k == maxit:
        print("Reached MAX Iter!")
    if full_time >= time_limit:
        print("Reached Time Limit")

    return x, {'k': k, 'f': f_tot_for_logs(x), 't': full_time, 'max_alpha': np.max(alpha), 'f_eval_sub_mean': np.mean(count), 'f_eval_sub': np.sum(count)}
