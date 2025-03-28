import numpy as np
import argparse

from problems import * 
from algorithms.DF_penalty_dec import DF_penalty_dec 
from algorithms.DF_penalty_dec_parallel import DF_penalty_dec_parallel

from log_utils import *

ALL_PROBLEMS = {
    'ARWHEAD': ARWHEAD,
}

ALL_DIMENSIONS = {
    'ARWHEAD': [10], 
}

def main(n, problem_name, exp_log_dir, algorithm, 
            pause, maxit, toll, time_limit, 
            df_pd_tau, df_pd_max_tau, df_pd_theta):
    
    np.random.seed(42)
        
    if problem_name in ALL_PROBLEMS:
        prob = ALL_PROBLEMS[problem_name]
        f_list, x0, I = prob(n, pause)

        if pause > 0: # pauseless way to compute the total objective
            f_list_for_logs, _, _ = prob(n, 0.)
            f_tot = lambda x: sum(f_i_logs(x) for f_i_logs in f_list_for_logs)
        else:
            f_tot = lambda x: sum(f(x) for f in f_list)
        
    else:
        raise ValueError("Invalid problem name")

    f_0 = f_tot(x0)
    m = len(f_list)

    if df_pd_tau is None: 
        df_pd_tau = f_0 / m * 1e-2
    
    if df_pd_max_tau is None:
        df_pd_max_tau = f_0 / m

    df_pd_settings = {
        'toll': toll,
        'maxit': maxit,
        'tau': df_pd_tau,
        'max_tau': df_pd_max_tau,
        'theta': df_pd_theta,
        'delta': 0.5,
        'gamma': 1e-6,
        'pause': pause,
        'time_limit': time_limit,
    }

    if algorithm in ['ALL', 'PDDF']:
        print(" Penalty-Decomposition Derivative-Free ")
        out_dir = os.path.join(exp_log_dir, 'PDDF')
        os.makedirs(out_dir, exist_ok=True)

        x, k, func_eval_count = DF_penalty_dec(f_list, x0, I, **df_pd_settings, f_tot_for_logs=f_tot, out_dir=out_dir)
        print(f"f: {f_tot(x)}    n_evals: {func_eval_count}    iters: {k} \n")
        
    if algorithm in ['ALL', 'PDDF_P']:
        print(" Penalty-Decomposition Derivative-Free [Parallel Implementation] ")
        out_dir = os.path.join(exp_log_dir, 'PDDF_P')
        os.makedirs(out_dir, exist_ok=True)

        x, k, func_eval_count = DF_penalty_dec_parallel(f_list, x0, I, **df_pd_settings, f_tot_for_logs=f_tot, out_dir=out_dir)
        print(f"f: {f_tot(x)}    n_evals: {func_eval_count}    iters: {k} \n")
       

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--problem_class', choices=['ALL', 'SMALL', 'LARGE'])
    parser.add_argument('-ld', '--logs_dir', required=True)
    parser.add_argument('-a', '--algorithm', choices=['ALL', 'PDDF', 'PDDF_P'])
    
    parser.add_argument('-pause', default=0., type=float)
    parser.add_argument('-toll', default=1e-4, type=float)
    parser.add_argument('-maxit', default=10000)
    parser.add_argument('-tl', '--time_limit', default=7200, type=float)

    parser.add_argument('-df_pd_tau', default=None, type=float) 
    parser.add_argument('-df_pd_max_tau', default=None, type=float) 

    parser.add_argument('-df_pd_theta', default=1.5, type=float)

    args = parser.parse_args()

    savedir = init_logs_folder(args.logs_dir)

    for problem, dimensions in ALL_DIMENSIONS.items():

        for n in dimensions:

            if args.problem_class == 'SMALL' and n > 700:
                continue
            if args.problem_class == 'LARGE' and n < 700:
                continue
        
            expdir = os.path.join(savedir, problem + ' ' + str(n))
            os.makedirs(expdir, exist_ok=True)

            print("\nPROBLEM:", problem, "   N:", n, "\n")

            main(n, problem, expdir, args.algorithm,
                 pause=args.pause, maxit=args.maxit, toll=args.toll, time_limit=args.time_limit,
                 df_pd_tau=args.df_pd_tau, df_pd_max_tau=args.df_pd_max_tau, df_pd_theta=args.df_pd_theta)
             