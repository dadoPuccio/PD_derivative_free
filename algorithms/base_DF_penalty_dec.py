import numpy as np
import os
import time
from log_utils import save_json, init_csv, append_row_csv

def DF_penalty_dec(f_list, x0, I, lb, ub,
                   tau, max_tau, toll, delta, gamma, theta, sigma, maxit, max_subf_eval,
                   pause, time_limit, f_tot_for_logs, out_dir, constr_policy):

    assert constr_policy in ['unconstr', 'internal', 'external']
    
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
                'constr_policy': constr_policy,
                'max_subf_eval': max_subf_eval})
    
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

    common_indices = {}
    for j in range(n):
        indices = [i for i, I_i in enumerate(I) if j in I_i]
        if len(indices) > 1:
            for i in indices:
                if i not in common_indices:
                    common_indices[i] = []
                common_indices[i].append(j)

    k = 0
    v = np.tile(x0[:, np.newaxis], q) # replicate x0, q times
    x = x0.copy()
    last_x = x.copy()
    termination = False

    count = np.zeros(q)
    
    e_time = time.time()
    full_time += e_time - s_time


    # LOGS are not considered on time eval
    col_names = ['k', 'f', 'time', 'norm_update', 'max_alpha', 'min_count', 'avg_count', 'sum_count', 'tau']
    init_csv(os.path.join(out_dir), 'log.csv', col_names)

    print('k     f     time      norm_update max_alpha min_count avg_count sum_count tau')
    print(f"{k}  {f_tot_for_logs(x):.6f} {full_time:.6f}  {0:.6f}  {np.max(alpha):.6f}    {int(np.min(count))}    {np.mean(count):.4f}   {int(np.sum(count))}   {tau:.4f}")

    out_row = [k, f_tot_for_logs(x), full_time, 0, np.max(alpha), np.min(count), np.mean(count), np.sum(count), tau] # + [f_i(x) for f_i in f_list] + [c for c in count]
    append_row_csv(os.path.join(out_dir), 'log.csv', out_row)


    while k < maxit and full_time < time_limit:

        s_time = time.time()

        prev_x = x.copy()

        for i in range(q):
            f_i = f_list[i]
            current_v = v[:, i].copy()
            
            def q_fun(a, b):
                return f_i(a) + 0.5 * tau * np.sum((a[I[i]] - b[I[i]]) ** 2)

            current_q_val = None
            
            for j in I[i]:
                
                if constr_policy == 'internal':
                    current_v, current_q_val, alpha_ij, count_i, n_exp = coordinate_search_constr(current_v, alpha[i, j], q_fun, x, count[i], gamma, delta, d[:, j], lb[j], ub[j], j, current_q_val=current_q_val)
                else:
                    current_v, current_q_val, alpha_ij, count_i, n_exp = coordinate_search(current_v, alpha[i, j], q_fun, x, count[i], gamma, delta, d[:, j], current_q_val=current_q_val)

                alpha[i][j] = alpha_ij
                count[i] = count_i + n_exp
            
            v[:, i] = current_v
        
        if constr_policy in ['external', 'internal']:
            x = exact_minimum_costr(I, v, n, lb, ub)
        else:
            x = exact_minimum(I, v, n)


        update_norm = np.linalg.norm(x - prev_x)
        max_alpha = np.max(alpha)
        
        if update_norm < toll * 100 and max_alpha < max((toll * 100) / max(1, tau), toll):
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
            print(f"{k}  {f_tot_for_logs(x):.6f} {full_time:.6f}  {update_norm:.6f}  {max_alpha:.6f}    {int(np.min(count))}    {np.mean(count):.4}   {int(np.sum(count))}   {tau:.4f}")
        
        out_row = [k, f_tot_for_logs(x), full_time, update_norm, max_alpha, np.min(count), np.mean(count),  np.sum(count),  tau]
        append_row_csv(os.path.join(out_dir), 'log.csv', out_row)

        if termination:
            # check_feasible(x, lb, ub)
            break
    
    if k == maxit:
        print("Reached MAX Iter!")
    if full_time >= time_limit:
        print("Reached Time Limit")
    
    return x, {'k': k, 'f': f_tot_for_logs(x), 't': full_time, 'max_alpha': np.max(alpha), 'f_eval_sub_mean': np.mean(count), 'f_eval_sub': np.sum(count)}

def expansion_step(q_fun, current_q_val, current_v, x, v_tent, q_tent, d, step, gamma, delta):
    v_next = v_tent.copy()
    q_next = q_tent

    v_tent = current_v + (step / delta) * d
    q_tent = q_fun(v_tent, x)
    n_exp = 1
    
    while q_tent <= current_q_val - gamma * ((step / delta) ** 2) and q_tent != current_q_val:
        step /= delta
        v_next = v_tent.copy()
        q_next = q_tent
        v_tent = current_v + (step / delta) * d
        q_tent = q_fun(v_tent, x)
        n_exp += 1
    
    return v_next, q_next, step, n_exp


def expansion_step_constr(q_fun, current_q_val, current_v, x, v_tent, q_tent, d, step, gamma, delta, j, bj, upper=True):

    alpha_acc = step

    if upper:
        assert bj >= current_v[j] - 1e-10
    else:
        assert bj <= current_v[j] + 1e-10

    max_alpha = (bj - current_v[j]) if upper else (current_v[j] - bj)
    alpha_tent = min(max_alpha, alpha_acc / delta)

    v_next = v_tent.copy()
    q_next = q_tent
    v_tent = current_v + alpha_tent * d
    q_tent = q_fun(v_tent, x)
    n_exp = 1
    
    while q_tent <= current_q_val - gamma * (alpha_tent ** 2) and q_tent != current_q_val:

        v_next = v_tent.copy()
        q_next = q_tent

        alpha_acc = alpha_tent
        alpha_tent = min(max_alpha, alpha_acc / delta)

        if alpha_tent == alpha_acc:
            break

        v_tent = current_v + alpha_tent * d
        q_tent = q_fun(v_tent, x)
        n_exp += 1
    
    return v_next, q_next, alpha_acc, n_exp


def exact_minimum(I, v, n):
    avg_v_val = np.zeros(n)
    for j in range(n):
        indices = [i for i, I_i in enumerate(I) if j in I_i]
        v_vals = v[j, indices]
        avg_v_val[j] = np.mean(v_vals)
    return avg_v_val

def exact_minimum_costr(I, v, n, lb, ub):
    avg_v_val = np.zeros(n)
    for j in range(n):
        indices = [i for i, I_i in enumerate(I) if j in I_i]
        v_vals = v[j, indices]
        # if len(v_vals) > 1:
        #     print(lb[j], np.mean(v_vals), v_vals, ub[j])
        avg_v_val[j] = max(min(np.mean(v_vals), ub[j]), lb[j])
    return avg_v_val

def update_alpha(I, alpha, n):
    for j in range(n):
        indices = [i for i, I_i in enumerate(I) if j in I_i]
        if len(indices) > 1:
            alpha_vals = alpha[indices, j]
            alpha[indices, j] = np.mean(alpha_vals)

    return alpha


def coordinate_search(current_v, alpha_ij, q_fun, x, count_i, gamma, delta, d_j, current_q_val=None):

    if alpha_ij > 0:

        if current_q_val is None:
            current_q_val = q_fun(current_v, x)
            count_i += 1

        v_tent = current_v + alpha_ij * d_j # positive direction
        q_tent = q_fun(v_tent, x)
        count_i += 1
        
        if q_tent <= current_q_val - gamma * (alpha_ij ** 2) and q_tent != current_q_val:
            next_v, next_q, a, n_exp = expansion_step(q_fun, current_q_val, current_v, x, v_tent, q_tent, d_j, alpha_ij, gamma, delta)
            alpha_ij = a
            current_v = next_v.copy()
            current_q_val = next_q

        else:
            v_tent = current_v - alpha_ij * d_j # negative direction
            q_tent = q_fun(v_tent, x)
            count_i += 1
            
            if q_tent <= current_q_val - gamma * (alpha_ij ** 2) and q_tent != current_q_val:
                next_v, next_q, a, n_exp = expansion_step(q_fun, current_q_val, current_v, x, v_tent, q_tent, - d_j, alpha_ij, gamma, delta)
                alpha_ij = a
                current_v = next_v.copy()
                current_q_val = next_q

            else:
                n_exp = 0
                alpha_ij *= delta
    else:
        n_exp = 0
        current_q_val = None

    return current_v, current_q_val, alpha_ij, count_i, n_exp
    

def coordinate_search_constr(current_v, alpha_ij, q_fun, x, count_i, gamma, delta, d_j, lb_j, ub_j, j, current_q_val=None):

    if alpha_ij > 0:
        
        if current_q_val is None:
            current_q_val = q_fun(current_v, x)
            count_i += 1

        alpha = min(ub_j - current_v[j], alpha_ij)
        
        if alpha > 1e-10:
            v_tent = current_v + alpha * d_j # positive direction
            q_tent = q_fun(v_tent, x)
            count_i += 1

        else:
            q_tent = current_q_val
        
        if q_tent <= current_q_val - gamma * (alpha ** 2) and q_tent != current_q_val:
            if alpha_ij == alpha: # the constraint is not active, I test if I can expand the step size
                next_v, next_q, a, n_exp = expansion_step_constr(q_fun, current_q_val, current_v, x, v_tent, q_tent, d_j, alpha, gamma, delta, j, ub_j, upper=True)
                alpha_ij = a
                current_v = next_v.copy()
                current_q_val = next_q
            else:
                n_exp = 0
                alpha_ij = alpha 
                current_v = v_tent.copy()
                current_q_val = q_tent

        else:
            
            alpha = min(current_v[j] - lb_j, alpha_ij)

            if alpha > 1e-10:
                v_tent = current_v - alpha * d_j # negative direction
                q_tent = q_fun(v_tent, x)
                count_i += 1
            
            else:
                q_tent = current_q_val
            
            if q_tent <= current_q_val - gamma * (alpha ** 2) and q_tent != current_q_val:
                if alpha_ij == alpha:
                    next_v, next_q, a, n_exp = expansion_step_constr(q_fun, current_q_val, current_v, x, v_tent, q_tent, -d_j, alpha, gamma, delta, j, lb_j, upper=False)
                    alpha_ij = a
                    current_v = next_v.copy()
                    current_q_val = next_q
                else:
                    n_exp = 0
                    alpha_ij = alpha # CHECK IF THIS IS USEFUL!
                    current_v = v_tent
                    current_q_val = q_tent

            else:
                n_exp = 0
                alpha_ij *= delta
    else:
        n_exp = 0
        current_q_val = None

    return current_v, current_q_val, alpha_ij, count_i, n_exp


def check_feasible(v, lb, ub):

    if (v - lb < 0).any():
        print(v)
        print(lb)
        print('lb violated')
        input()
        return False
    if (ub - v < 0).any():
        print(v)
        print(ub)
        print('ub violated')
        return False
    return True
