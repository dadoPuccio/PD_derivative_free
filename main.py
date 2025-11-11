import numpy as np
import argparse

from algorithms.PDDF import PDDF
from algorithms.PDDF_P import PDDF_P
from algorithms.LS import line_search
from algorithms.SALS import structure_aware_LS
from  algorithms.MADS import mads_solve

from log_utils import *
from problems.problem_factory import *

def main(n, problem_name, exp_log_dir, savedir, algorithm, pause,
            maxit, max_eval_group_k, toll, time_limit, df_pd_tau, df_pd_max_tau, df_pd_theta):
    
    f_list, x0, I, lb, ub, f_tot, f_tot_for_logs = get_problem(problem_name, n, pause)

    if n is None:
        n = len(x0)

    max_eval = max_eval_group_k * (len(x0) + 1)
    max_subf_eval = max_eval_group_k * (len(x0) + 1) * len(f_list)

    if prob_name in LOCKWOOD_PROBLEM_NAMES:
        max_eval *= 100
        max_subf_eval *= 100  

    print("Max eval", max_eval)
    print("Max subf eval", max_subf_eval)

    ls_settings = {
        'toll': toll,
        'maxit': maxit,
        'delta': 0.5,
        'gamma': 1e-6,
        'pause': pause,
        'time_limit': time_limit,
    }
    
    pddf_settings = {
        'toll': toll,
        'maxit': maxit,
        'tau': df_pd_tau,
        'max_tau': df_pd_max_tau,
        'delta': 0.5,
        'gamma': 1e-6,
        'theta': df_pd_theta, 
        'sigma': 0.5,
        'pause': pause,
        'time_limit': time_limit,
        'constr_policy': 'unconstr' if lb is None and ub is None else 'internal',
        'max_iter_pd': 100 if problem_name not in LOCKWOOD_PROBLEM_NAMES else 10, 
        'toll_pd': 1e-2 if problem_name not in LOCKWOOD_PROBLEM_NAMES else 1e-1
    }

    if algorithm in ['ALL', 'STD', 'LS']:

        print(" Line-Search ")
        out_dir = os.path.join(exp_log_dir, 'LS')
        os.makedirs(out_dir, exist_ok=True)

        x, result = line_search(f_tot, x0, lb=lb, ub=ub, **ls_settings, out_dir=out_dir, n_functions=len(f_list), max_eval=max_eval)
        # print(f"x: {x}")
        print(f"f: {result['f']}    n_evals_sub: {result['f_eval_sub']}    iters: {result['k']} \n")

        out_result = {'problem': problem_name, 'n': n, 'max_subf_eval': max_subf_eval, 'time_limit': time_limit} | result
        if not os.path.exists(os.path.join(savedir, 'LS.csv')):
            init_csv(savedir, 'LS.csv', out_result.keys())
        
        append_row_csv(savedir, 'LS.csv', out_result.values())


    if algorithm in ['ALL', 'FAST', 'SALS']:
    
        print(" Structure Aware Line-Search ")
        out_dir = os.path.join(exp_log_dir, 'SALS')
        os.makedirs(out_dir, exist_ok=True)

        x, result = structure_aware_LS(f_list, I, x0, lb=lb, ub=ub, **ls_settings, out_dir=out_dir, max_subf_eval=max_subf_eval)
        # print(f"x: {x}")
        print(f"f: {result['f']}    time: {result['t']}    iters: {result['k']} \n")

        out_result = {'problem': problem_name, 'n': n, 'max_subf_eval': max_subf_eval, 'time_limit': time_limit} | result
        if not os.path.exists(os.path.join(savedir, 'SALS.csv')):
            init_csv(savedir, 'SALS.csv', out_result.keys())
        
        append_row_csv(savedir, 'SALS.csv', out_result.values())


    if algorithm in ['ALL', 'STD', 'PDDF']:

        print(" Penalty-Decomposition Derivative-Free ")
        out_dir = os.path.join(exp_log_dir, 'PDDF')
        os.makedirs(out_dir, exist_ok=True)

        x, result = PDDF(f_list, f_tot, x0, I, lb=lb, ub=ub, **pddf_settings, 
                         f_tot_for_logs=f_tot_for_logs, out_dir=out_dir, 
                         max_eval=max_eval, max_subf_eval=max_subf_eval)
        # print(f"x: {x}")
        print(f"f: {result['f']}    n_evals_sub: {result['f_eval_sub']}    iters: {result['k']} \n")

        out_result = {'problem': problem_name, 'n': n, 'max_subf_eval': max_subf_eval, 'time_limit': time_limit} | result
        if not os.path.exists(os.path.join(savedir, 'PDDF.csv')):
            init_csv(savedir, 'PDDF.csv', out_result.keys())
        
        append_row_csv(savedir, 'PDDF.csv', out_result.values())

    
    if algorithm in ['ALL', 'FAST', 'PDDF_P']:

        print(" Penalty-Decomposition Derivative-Free [Parallel Implementation]")
        out_dir = os.path.join(exp_log_dir, 'PDDF_P')
        os.makedirs(out_dir, exist_ok=True)

        x, result = PDDF_P(f_list, x0, I, lb=lb, ub=ub, **pddf_settings, f_tot_for_logs=f_tot_for_logs, out_dir=out_dir, 
                           max_eval=max_eval, max_subf_eval=max_subf_eval)
        # print(f"x: {x}")
        print(f"f: {result['f']}    time: {result['t']}    iters: {result['k']} \n")

        out_result = {'problem': problem_name, 'n': n, 'max_subf_eval': max_subf_eval, 'time_limit': time_limit} | result
        if not os.path.exists(os.path.join(savedir, 'PDDF_P.csv')):
            init_csv(savedir, 'PDDF_P.csv', out_result.keys())
        
        append_row_csv(savedir, 'PDDF_P.csv', out_result.values())


    if algorithm in ['ALL', 'STD', 'MADS']:
        
        print(" MADS SOLVE ")
        out_dir = os.path.join(exp_log_dir, 'MADS')
        os.makedirs(out_dir, exist_ok=True)

        x, result = mads_solve(f_tot, x0, lb, ub, len(f_list), f_tot_for_logs, pause, toll, time_limit, max_eval, out_dir=out_dir)
        # print(f"x: {x}")
        print(f"f: {result['f']}    f_eval_sub: {result['f_eval_sub']} \n")

        out_result = {'problem': problem_name, 'n': n, 'max_subf_eval': max_subf_eval, 'time_limit': time_limit} | result
        if not os.path.exists(os.path.join(savedir, 'MADS.csv')):
            init_csv(savedir, 'MADS.csv', out_result.keys())
        
        append_row_csv(savedir, 'MADS.csv', out_result.values())
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--problem_class', choices=['ALL', 'UNCONSTR', 'BOUNDED', 'LOCKWOOD'])
    parser.add_argument('-ld', '--logs_dir', required=True)
    parser.add_argument('-a', '--algorithm', choices=['ALL', 'STD', 'FAST', 'PDDF', 'PDDF_P', 'LS', 'SALS', 'MADS'])
    
    parser.add_argument('-pause', default=0., type=float)
    parser.add_argument('-toll', default=1e-4, type=float)
    parser.add_argument('-maxit', default=1e9, type=int)
    parser.add_argument('-mg', '--max_eval_group_k', default=1000, type=int)
    parser.add_argument('-tl', '--time_limit', default=600, type=int)

    parser.add_argument('-df_pd_tau', default=1., type=float)
    parser.add_argument('-df_pd_max_tau', default=1e8, type=float)

    parser.add_argument('-df_pd_theta', default=1.1, type=float)

    args = parser.parse_args()

    savedir = init_logs_folder(args.logs_dir)

    for problem, dimensions in ALL_DIMENSIONS.items():

        print(problem)

        if '_COPY' in problem:
            prob_name = problem.split('_')[0]
        else:
            prob_name = problem

        if args.problem_class == 'UNCONSTR' and prob_name not in UNCONSTR_PROBS_NAMES:
            continue

        if args.problem_class == 'BOUNDED' and prob_name not in BOUNDED_PROBS_NAMES:
            continue

        if args.problem_class == 'LOCKWOOD' and prob_name not in LOCKWOOD_PROBLEM_NAMES:
            continue

        for n in dimensions:
        
            expdir = os.path.join(savedir, problem + ' ' + str(n))
            os.makedirs(expdir, exist_ok=True)

            print("\nPROBLEM:", problem, "   N:", n, "\n")

            main(n, problem, expdir, savedir, args.algorithm,
                 pause=args.pause, maxit=args.maxit, max_eval_group_k=args.max_eval_group_k,
                 toll=args.toll, time_limit=args.time_limit,
                 df_pd_tau=args.df_pd_tau, df_pd_max_tau=args.df_pd_max_tau, df_pd_theta=args.df_pd_theta)
            