import os
from log_utils import save_json
from algorithms.base_DF_penalty_dec_parallel import DF_penalty_dec_parallel
from algorithms.SALS import structure_aware_LS

def PDDF_P(f_list, x0, I, lb, ub,
                tau, max_tau, toll, delta, gamma, theta, sigma, maxit, max_subf_eval, max_eval, max_iter_pd, toll_pd,
                pause, time_limit, f_tot_for_logs, out_dir, constr_policy):

    x, result_dfpd = DF_penalty_dec_parallel(f_list=f_list, x0=x0, I=I, lb=lb, ub=ub,
                                                tau=tau, max_tau=max_tau, toll=toll_pd, delta=delta, gamma=gamma, theta=theta,
                                                sigma=sigma, maxit=max_iter_pd, max_subf_eval=max_subf_eval,
                                                pause=pause, time_limit=time_limit, f_tot_for_logs=f_tot_for_logs, out_dir=out_dir, constr_policy=constr_policy)

    x, result_ls = structure_aware_LS(f_list, I=I, x0=x, lb=lb, ub=ub, toll=toll, delta=delta, gamma=gamma, for_hybrid=True,
                                        maxit=maxit - result_dfpd['k'], max_subf_eval=max_eval - int(result_dfpd['f_eval_sub']), pause=pause, time_limit=time_limit - result_dfpd['t'], out_dir=out_dir,
                                        hyb_k=result_dfpd['k'], hyb_time=result_dfpd['t'], hyb_eval_count=int(result_dfpd['f_eval_sub_mean']), hyb_subf_count=result_dfpd['f_eval_sub'], 
                                        hyb_alpha=min(result_dfpd['max_alpha'] * 10, 1))
    
    result = {'k': result_dfpd['k'] + result_ls['k'],
              'f': result_ls['f'], 
              't': result_dfpd['t'] + result_ls['t'], 
              'max_alpha': result_ls['max_alpha'], 
              'f_eval': int(result_dfpd['f_eval_sub_mean']) + result_ls['f_eval_sub_mean'], 
              'f_eval_sub': result_dfpd['f_eval_sub'] + result_ls['f_eval_sub'], }
    
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
                'max_subf_eval': max_subf_eval,
                'max_eval': max_eval})
    
    return x, result
 