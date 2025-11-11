import numpy as np
import PyNomad
from log_utils import save_json, init_csv, append_row_csv
import os
import time
import socket

def mads_solve(f_tot, x0, lb, ub, n_functions,
               f_tot_for_logs, pause, toll, time_limit, max_eval, out_dir):
    
    # https://github.com/bbopt/nomad/blob/master/examples/advanced/library/PyNomad/simpleExample_basic.py

    save_json(os.path.join(out_dir, 'params.json'), 
               {'toll': toll,
                'max_eval': max_eval,
                'pause': pause,
                'time_limit': time_limit})
    
    col_names = ['f', 'time', 'count', 'sum_subf_count'] 
    init_csv(os.path.join(out_dir), 'log.csv', col_names)

    out_row = [f_tot_for_logs(x0), 0, 0, 0] 
    append_row_csv(os.path.join(out_dir), 'log.csv', out_row)

    def bb(x):
        n = x.size()
        curr_sol = np.zeros((n,))
        for i in range(n):
            curr_sol[i] = x.get_coord(i)
        
        x.setBBO(str(f_tot(curr_sol)).encode("UTF-8"))
        return 1 # 1: success 0: failed evaluation
   
    min_frame_size_param = str(toll) 

    params = ["BB_OUTPUT_TYPE OBJ", 
              "MAX_BB_EVAL " + str(max_eval), 
              "MAX_TIME " + str(time_limit), 
              "MIN_FRAME_SIZE * " + min_frame_size_param,
              "DISPLAY_DEGREE 0",
              "DISPLAY_ALL_EVAL false",
              "STATS_FILE mads_stats_" + socket.gethostname() + ".txt BBE OBJ"
              ]
        
    if lb is None:
        lb = []
    else:
        lb = [v.item() for v in lb]

    if ub is None:
        ub = []
    else:
        ub = [v.item() for v in ub]

    full_time = 0.
    s_time = time.time()

    result = PyNomad.optimize(bb, x0, lb, ub, params)

    e_time = time.time()
    full_time += e_time - s_time

    print(result['stop_reason'])
    append_row_csv(out_dir, 'log.csv', [result['f_best'], full_time, result['nb_evals'], result['nb_evals'] * n_functions])

    convert_mads_stat_to_our("mads_stats_" + socket.gethostname() + ".0.txt", out_dir, n_functions)

    x = np.array(result['x_best'])
    return x, {'f': f_tot_for_logs(x), 't': full_time, 'f_eval': result['nb_evals'], 'f_eval_sub': result['nb_evals'] * n_functions}


def convert_mads_stat_to_our(mads_path, out_dir, n_functions):

    data = np.loadtxt(mads_path)
    init_csv(os.path.join(out_dir), 'log_mads.csv', ['k', 'f', 'count', 'sum_subf_count'])

    if len(data.shape) == 1:
        append_row_csv(out_dir, 'log_mads.csv', [data[0], data[1], data[0], data[0] * n_functions])

    else:

        for k, row in enumerate(data):

            append_row_csv(out_dir, 'log_mads.csv', [k, row[1], row[0], row[0] * n_functions])
