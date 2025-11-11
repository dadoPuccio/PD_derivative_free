import numpy as np

from problems.problems import *
from problems.original_problems import *

ORIGINAL_PROBLEMS = {
    'ARWHEAD': ARWHEAD,
    'BDQRTIC': BDQRTIC,
    'BEALES': BEALES,
    'BROYDN3D': BROYDN3D,
    'BROWNAL6': BROWNAL6,
    'DIXMAANA': DIXMAANA,
    'DIXMAANI': DIXMAANI,
    'ENGVAL': ENGVAL,
    'FREUROTH': FREUROTH,
    'MOREBV': MOREBV,
    'NZF1': NZF1,
    'POWSING': POWSING,
    'ROSENBR': ROSENBR,
    'TRIDIA': TRIDIA,
    'WOODS': WOODS_NEW,
}

ALL_DIMENSIONS = {
    'ARWHEAD': [10, 50, 100, 500, 1000],  
    'BDQRTIC': [10, 50, 100, 500, 1000], 
    'BEALES': [10, 50, 100, 500, 1000],      
    'BROWNAL6': [10, 50, 102, 502], 
    'BROYDN3D': [10, 50, 100, 500, 1000],   
    'DIXMAANA': [15, 51, 102, 501], 
    'DIXMAANI': [15, 51, 102, 501], 
    'ENGVAL': [10, 50, 100, 500, 1000],    
    'FREUROTH': [10, 50, 100, 500, 1000],   
    'MOREBV': [12, 52, 102, 502],
    'NZF1': [13, 39, 130, 650],
    'POWSING': [20, 52, 100, 500],  
    'ROSENBR': [10, 50, 100, 500, 1000],   
    'TRIDIA': [10, 50, 100, 500, 1000],    
    'WOODS': [20, 40, 200, 400],
}

UNCONSTR_CUTEST_ORIG_DIMS = { 
    # # CUTEST UNCONSTRAINED PROBLEMS WITH ORIGINAL DIMENSION BETWEEN 2 AND 100
    'AKIVA': 2, #academic
    'BA-L1SPLS': 57, #modelling
    'BARD': 3, #academic
    'BEALE': 2, #academic
    'BIGGS6': 6, #academic
    'BOXBODLS': 2, #modelling
    'BRKMCC': 2, #academic
    'CHNROSNB': 50, #academic
    'CHNRSNBM': 50, #academic
    'CLUSTERLS': 2, #academic
    'COOLHANSLS': 9, #real-world
    'DENSCHNA': 2, #academic
    'DENSCHNB': 2, #academic
    'DENSCHND': 3, #academic
    'DEVGLA1': 4, #modelling
    'ERRINROS': 50, #academic
    'ERRINRSM': 50, #academic
    'EXPFIT': 2, #academic
    'HATFLDFLS': 3, #academic
    'HATFLDGLS': 25, #academic
    'HILBERTA': 2, #academic
    'HILBERTB': 10, #academic
    'HIMMELBCLS': 2, #academic
    'HIMMELBG': 2, #academic
    'LRIJCNN1': 22, #academic
    'LUKSAN11LS': 100, #academic
    'LUKSAN12LS': 98, #academic
    'LUKSAN13LS': 98, #academic
    'LUKSAN14LS': 98, #academic
    'LUKSAN21LS': 100, #academic
    'POWERSUM': 4, #modelling
    'QING': 100, #modelling
    'SNAIL': 2, #academic
    'SSI': 3, #academic
    'TESTQUAD': 10, #academic
    'TOINTGOR': 50, #modelling
    'TOINTPSP': 50, #academic
    'TOINTQOR': 50, #modelling
    'TRIGON1': 10, #modelling
    'VANDANMSLS': 22, #modelling
    'WATSON': 12, #academic
}

UNCONSTR_COMPOSITE = {
    'AKIVA_BEALE_BOXBODLS_BRKMCC_BROWNBS_1_COMP': [None],
    'BA-L1LS_BA-L1SPLS_8_COMP': [None],
    'CHNROSNB_CHNRSNBM_8_COMP': [None],
    'CHWIRUT1LS_CHWIRUT2LS_2_COMP': [None],
    'TOINTGOR_TOINTPSP_TOINTQOR_8_COMP': [None],
    'LUKSAN11LS_LUKSAN12LS_LUKSAN13LS_LUKSAN14LS_LUKSAN15LS_LUKSAN16LS_LUKSAN17LS_LUKSAN21LS_LUKSAN22LS_16_COMP': [None], 
    'DENSCHNA_DENSCHNB_DENSCHNC_DENSCHND_DENSCHNE_DENSCHNF_1_COMP': [None],
    'LANCZOS1LS_LANCZOS2LS_LANCZOS3LS_2_COMP': [None],
    'VESUVIALS_VESUVIOLS_VESUVIOULS_2_COMP': [None],
    'ALLINITU_BROWNDEN_BIGGS6_FBRAIN3LS_STREG_2_COMP': [None],
    'DIAMON2DLS_DIAMON3DLS_8_COMP': [None],
    'TRIGON1_TRIGON2_2_COMP': [None],
    'BARD_BOX3_CLIFF_CLUSTERLS_CUBE_DJTL_ENGVAL2_EXPFIT_GAUSSIAN_GROWTHLS_HAIRY_1_COMP': [None], 
    'GAUSS1LS_GAUSS2LS_GAUSS3LS_2_COMP': [None],
    'MISRA1ALS_MISRA1BLS_MISRA1CLS_MISRA1DLS_1_COMP': [None],
    'HYDC20LS_HYDCAR6LS_16_COMP': [None],
    'MGH09LS_MGH10LS_MGH10SLS_MGH17LS_MGH17SLS_NELSONLS_OSBORNEA_OSBORNEB_2_COMP': [None],
    'MANCINO_QING_SENSORS_32_COMP': [None],
    'HEART6LS_HEART8LS_2_COMP': [None]
}

UNCONSTR_PROBS_NAMES = [k for k in ORIGINAL_PROBLEMS] + [k for k in UNCONSTR_CUTEST_ORIG_DIMS] + [k for k in UNCONSTR_COMPOSITE]

BOUNDED_CUTEST_ORIG_DIMS = { 
    'DGOSPEC': 3, #academic
    'DIAGIQB': 10, #academic
    'HATFLDA': 4, #academic
    'HATFLDB': 4, #academic
    'DEVGLA1B': 4, #modelling
    'HS110': 10, #academic
    'POWERSUMB': 4, #modelling
    'QINGB': 5, #modelling
    'TRIGON1B': 10, #modelling
}

BOUNDED_COMPOSITE = {
    'BQPGABIM_BQPGASIM_8_C3': [None],
    'DEVGLA1B_DEVGLA2B_2_C3': [None],
    'FBRAIN2LS_FBRAINLS_2_C3': [None],
    'HATFLDA_HATFLDB_HATFLDC_4_C3': [None],
    'HS1_HS110_HS2_HS25_HS3_HS38_HS3MOD_HS4_HS5_1_C3': [None],
    'PFIT1LS_PFIT2LS_PFIT3LS_PFIT4LS_2_C3': [None],
    'TRIGON1B_TRIGON2B_4_C3': [None],
    '3PK_CHEBYQAD_DECONVB_SANTALS_8_C3': [None]
}

BOUNDED_PROBS_NAMES = [k for k in BOUNDED_CUTEST_ORIG_DIMS] + [k for k in BOUNDED_COMPOSITE]

MAX_DIMENSION = 1000
N_VERSIONS = 4

for p, d in UNCONSTR_CUTEST_ORIG_DIMS.items():
    ALL_DIMENSIONS[p + '_COPY'] = [(d - 1) * (2**i) + 1 for i in range(2, N_VERSIONS + 2) if (d - 1) * (2**i) + 1 < MAX_DIMENSION]

for p, d in BOUNDED_CUTEST_ORIG_DIMS.items():
    ALL_DIMENSIONS[p + '_COPY'] = [(d - 1) * (2**i) + 1 for i in range(2, N_VERSIONS + 2) if (d - 1) * (2**i) + 1 < MAX_DIMENSION]

ALL_DIMENSIONS.update(UNCONSTR_COMPOSITE)
ALL_DIMENSIONS.update(BOUNDED_COMPOSITE)

ALL_DIMENSIONS['LOCKWOOD1'] = [6]
ALL_DIMENSIONS['LOCKWOOD2'] = [21] 

LOCKWOOD_PROBLEM_NAMES = ['LOCKWOOD1', 'LOCKWOOD2'] 


def get_problem(problem_name, n, pause):
    np.random.seed(42)
    
    if 'LOCKWOOD' in problem_name:
        if '1' in problem_name:
            f_list, x0, I, lb, ub = LOCKWOOD1(n, pause)

            if pause > 0:
                f_list_for_logs, lb, ub = LOCKWOOD1(n, 0.)
                f_tot_for_logs = lambda x: sum(f_i_logs(x) for f_i_logs in f_list_for_logs)
            else:
                f_tot_for_logs = lambda x: sum(f(x) for f in f_list) 

        elif '2' in problem_name:
            f_list, x0, I, lb, ub = LOCKWOOD2(n, pause)

            if pause > 0:
                f_list_for_logs, lb, ub = LOCKWOOD2(n, 0.)
                f_tot_for_logs = lambda x: sum(f_i_logs(x) for f_i_logs in f_list_for_logs)
            else:
                f_tot_for_logs = lambda x: sum(f(x) for f in f_list) 
        
    elif '_' in problem_name:

        splitted_prob_name = problem_name.split("_")

        if len(splitted_prob_name) == 2:
            prob_name = splitted_prob_name[0]
            
            if prob_name in UNCONSTR_CUTEST_ORIG_DIMS.keys():
                constrained = False
            elif prob_name in BOUNDED_CUTEST_ORIG_DIMS.keys():
                constrained = True
            else:
                raise ValueError("Invalid problem name")
        
            f_list, x0, I, lb, ub = CUTEST(n=n, prob_name=prob_name, constrained=constrained, pause=pause)

            if pause > 0:
                f_list_for_logs, _, _, _, _ = CUTEST(n=n, prob_name=prob_name, constrained=constrained, pause=0.)
                f_tot_for_logs = lambda x: sum(f_i_logs(x) for f_i_logs in f_list_for_logs)
            else:
                f_tot_for_logs = lambda x: sum(f(x) for f in f_list)

        elif len(splitted_prob_name) > 2:

            if problem_name in UNCONSTR_COMPOSITE:
                constrained = False
            elif problem_name in BOUNDED_COMPOSITE:
                constrained = True
            else:
                raise ValueError("Invalid problem name")
            
            list_of_prob_names = splitted_prob_name[:-2]
            n_shared = int(splitted_prob_name[-2])

            f_list, x0, I, lb, ub = CUTEST_COMPOSITE(list_of_prob_names, n_shared, constrained=constrained, pause=pause)
            
            if pause > 0:
                f_list_for_logs, _, _, _, _ = CUTEST_COMPOSITE(list_of_prob_names, n_shared, constrained=constrained, pause=0.)
                f_tot_for_logs = lambda x: sum(f_i_logs(x) for f_i_logs in f_list_for_logs)
            else:
                f_tot_for_logs = lambda x: sum(f(x) for f in f_list)

    else:

        f_list, x0, I = ORIGINAL_PROBLEMS[problem_name](n=n, pause=pause)

        if pause > 0:
            f_list_for_logs, _, _ = ORIGINAL_PROBLEMS[problem_name](n=n, pause=0.)
            f_tot_for_logs = lambda x: sum(f_i_logs(x) for f_i_logs in f_list_for_logs)
        else:
            f_tot_for_logs = lambda x: sum(f(x) for f in f_list)
        
        lb, ub = None, None
        
    f_tot = lambda x: sum(f(x) for f in f_list) # Overall function

    return f_list, x0, I, lb, ub, f_tot, f_tot_for_logs
