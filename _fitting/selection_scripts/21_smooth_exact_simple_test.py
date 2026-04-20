import multiprocessing
import sys
from pathlib import Path
project_root = Path.cwd()  # importing functions from other folders
sys.path.insert(0, str(project_root))

import os
import random
from multiprocessing import Pool
from _data.data_utils import read_in
from _fitting.model_utils import data_settings_to_name, compare_models
from _fitting.model_utils_smooth import *
import arviz as az
import pandas as pd
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="arviz")

az.style.use("arviz-darkgrid")

################################################################################
if '___laptop' in os.listdir('.'):
    # laptop folder
    folder = "../_data/p-dengue/"
    outpath = "../_data/p-dengue/model_fits/"
elif '___server' in os.listdir('.'):
    # server folder
    folder = "../../../../data/lucaratzinger_data/p_dengue/"
    outpath = "../../../../data/lucaratzinger_data/p_dengue/model_fits"
else:
    print('something wrong')

################################################################################
data_settings = {'admin':2, 'max_lag':6, 'start_year':2016, 'start_month':1, 'end_year':2019, 'end_month':12}
data_name = data_settings_to_name(data_settings)

fitting_task = '21_smooth_exact_simple_test(temperature_higher_samples)'

################################################################################

def init_worker():
    """Initialize worker process with data"""
    global worker_data, worker_data_name, worker_outpath
    global _first_task, _worker_id
    _first_task = True
    _worker_id = multiprocessing.current_process()._identity[0]
    worker_data = read_in(folder, **data_settings, standardise=True, dropna=True, celsius=True, tp_log=True)
    worker_data_name = data_name
    worker_outpath = outpath
    print(f"Worker initialized with data: {worker_data_name}")

def worker(task):
    global _first_task, _worker_id
    if _first_task:
        print('stutter')
        # time.sleep(_worker_id * 30)

    """Fit a single model"""
    model_name, model_settings = task
    print(f'Fitting model: {model_name}')
    try:
        fit_sig_spline_p_model(
            worker_data, 
            worker_data_name, 
            model_settings, 
            worker_outpath,
            fitting_task,
            n_chains=4, 
            n_draws=5000, 
            n_tune=1000,
            sampler='nutpie',
            invert_log=True,
            centred_w=True,
            check_report=True,
            check_idata=True,
            clear_idata=False,
            basis_scale=1,
            model_builder='build_model_exact_mult_stat',
            show = {'summary': True, 'trace': False, 'pair': False, 'metrics': True,
                    'spline': True, 'exp_spline': True, 'link': False, 'link_spline': False,
                    'divergences': False}
        )
        _first_task = False
        return (model_name, "success")
        
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")
        _first_task = False
        return (model_name, f"failed: {e}")

################################################################################
_data = read_in(folder, **data_settings, standardise=True, dropna=True, celsius=True, tp_log=True)
statistics = _data.columns.tolist()
# that start with t2m, rh, tp
statistics = [name for name in statistics if name.startswith(('t2m','rh','tp'))]
# that contains 'pop_weighted'
statistics = [name for name in statistics if 'pop_weighted' in name]
# if it contains 'tp' then it should contain 'log('
statistics = [name for name in statistics if not (name.startswith('tp') and 'log(' not in name)]
lags = [0, 1, 2, 3, 4, 5, 6]
statistics = [stat for stat in statistics if any(f"({lag})" in stat for lag in lags)]

print(len(statistics))
print(statistics)
################################################################################
#p_vals = [1.0, 16.0]
#num_knots_list = [30]
#statistics = ['rh_mean_pop_weighted(0)', 't2m_mean_pop_weighted(0)', 't2m_min_pop_weighted(3)', 'tp_24hmax_pop_weighted_log(5)']

# p_vals = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 16.0, 20.0, 25.0, 1000.0]
# num_knots_list = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]

# p_vals = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 16.0, 25.0, 1000.0]
# p_vals = [0.05, 0.1, 0.25, 0.5]
# num_knots_list = [3, 5, 10, 15, 20, 30, 35, 40]

# p_vals = [1.0, 2.5, 5.0, 10.0, 16.0, 25.0, 1000.0]
# num_knots_list = [3]

#p_vals = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
#num_knots_list = [2, 3, 5, 10, 15]
#statistics = ['rh_mean_pop_weighted(0)']


selected_complete_simple = { # adjusted for pareto k and sampling
    'rh_mean_pop_weighted(0)': [(2.5, 5)],
    'rh_mean_pop_weighted(1)': [(2.5, 5)],
    'rh_mean_pop_weighted(2)': [(2.5, 5)],
    'rh_mean_pop_weighted(3)': [(2.5, 5)],
    'rh_mean_pop_weighted(4)': [(2.5, 5)],
    'rh_mean_pop_weighted(5)': [(2.5, 5)],
    'rh_mean_pop_weighted(6)': [(2.5, 5)],
    ###
    't2m_max_pop_weighted(0)': [(2.5, 20)],
    't2m_max_pop_weighted(1)': [(2.5, 20)],
    't2m_max_pop_weighted(2)': [(2.5, 20)],
    't2m_max_pop_weighted(3)': [(2.5, 20)],
    't2m_max_pop_weighted(4)': [(2.5, 20)],
    't2m_max_pop_weighted(5)': [(2.5, 20)],
    't2m_max_pop_weighted(6)': [(2.5, 20)],
    ###
    't2m_mean_pop_weighted(0)': [(5.0, 10)],
    't2m_mean_pop_weighted(1)': [(5.0, 10)],
    't2m_mean_pop_weighted(2)': [(5.0, 10)],
    't2m_mean_pop_weighted(3)': [(5.0, 10)],
    't2m_mean_pop_weighted(4)': [(5.0, 10)],
    't2m_mean_pop_weighted(5)': [(5.0, 10)],
    't2m_mean_pop_weighted(6)': [(5.0, 10)],
    ###
    't2m_min_pop_weighted(0)': [(2.5, 20)],
    't2m_min_pop_weighted(1)': [(2.5, 20)],
    't2m_min_pop_weighted(2)': [(2.5, 20)],
    't2m_min_pop_weighted(3)': [(2.5, 20)],
    't2m_min_pop_weighted(4)': [(2.5, 20)],
    't2m_min_pop_weighted(5)': [(2.5, 20)],
    't2m_min_pop_weighted(6)': [(2.5, 20)],
    ###
    'tp_24hmax_pop_weighted_log(0)': [(2.5, 5)],
    'tp_24hmax_pop_weighted_log(1)': [(2.5, 5)],
    'tp_24hmax_pop_weighted_log(2)': [(2.5, 5)],
    'tp_24hmax_pop_weighted_log(3)': [(2.5, 5)],
    'tp_24hmax_pop_weighted_log(4)': [(2.5, 5)],
    'tp_24hmax_pop_weighted_log(5)': [(2.5, 5)],
    'tp_24hmax_pop_weighted_log(6)': [(2.5, 5)],
    ###
    'tp_24hmean_pop_weighted_log(0)': [(2.5, 5)],
    'tp_24hmean_pop_weighted_log(1)': [(2.5, 5)],
    'tp_24hmean_pop_weighted_log(2)': [(2.5, 5)],
    'tp_24hmean_pop_weighted_log(3)': [(2.5, 5)],
    'tp_24hmean_pop_weighted_log(4)': [(2.5, 5)],
    'tp_24hmean_pop_weighted_log(5)': [(2.5, 5)],
    'tp_24hmean_pop_weighted_log(6)': [(2.5, 5)],
}

#selected_complete_simple = {s:[(2.5, 5)] for s in statistics}
p = {s: vals[0][0] for s, vals in selected_complete_simple.items()}
num_knots = {s: vals[0][1] for s, vals in selected_complete_simple.items()}

# greedy_3
#1
# s1 = ['tp_24hmean_pop_weighted_log(1)', 't2m_max_pop_weighted(4)']
#2
# s1 = ['tp_24hmean_pop_weighted_log(1)', 't2m_mean_pop_weighted(4)']
#3
# s1 = ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)']

# greedy_3
#1
# s1 = ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)']
#2
# s1 = ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)']
#3
# s1 = ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)']

#for s in s1:
    #statistics.remove(s)
profiles = [['tp_24hmean_pop_weighted_log(1)', 't2m_max_pop_weighted(4)', 'tp_24hmean_pop_weighted_log(6)'],
            ['tp_24hmean_pop_weighted_log(1)', 't2m_max_pop_weighted(4)', 't2m_min_pop_weighted(2)'],
            ['tp_24hmean_pop_weighted_log(1)', 't2m_max_pop_weighted(4)', 't2m_mean_pop_weighted(0)'],

            ['tp_24hmean_pop_weighted_log(1)', 't2m_mean_pop_weighted(4)', 'tp_24hmean_pop_weighted_log(6)'],
            ['tp_24hmean_pop_weighted_log(1)', 't2m_mean_pop_weighted(4)', 't2m_min_pop_weighted(6)'],
            ['tp_24hmean_pop_weighted_log(1)', 't2m_mean_pop_weighted(4)', 'tp_24hmax_pop_weighted_log(0)'],

            ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)', 't2m_mean_pop_weighted(1)'],
            ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)', 't2m_max_pop_weighted(6)'],
            ['rh_mean_pop_weighted(1)', 'tp_24hmean_pop_weighted_log(5)', 't2m_mean_pop_weighted(3)']]
print(len(list(set(tuple(x) for x in profiles))))

#p_vals = [0.05]
#num_knots_list = [15]
#statistics = ['t2m_max_pop_weighted(0)']

if __name__ == "__main__":
    # Build model dictionary
    model_dict = {}
    for s1 in profiles:
            stat_names = s1
            settings = {
                'alpha_type': 'exponential', 'alpha_parameters': {'lam': 0.5},
                'intercept_type': 'normal', 'intercept_parameters': {'mu': -10.0, 'sigma': 1.0},
                'beta_u_type': 'normal', 'beta_u_parameters': {'mu': 0, 'sigma': 1.0},
                'link': None, 'link_stat_name': None, 'link_type': None,
                'b1_type': None, 'b1_parameters': None,
                'c_type': None, 'c_parameters': None,
                'stat_names': stat_names, 'num_knots': num_knots, 'knot_type': 'equispaced', 'degree': 3,
                'spline_implementation': 'svd', 'spline_type': 'halfnormal', 'spline_parameters': {'sigma_w_sigma': 10.0},
                'penalty_order': 2, 'penalty_type': 'halfnormal', 'penalty_parameters': {'p': p}, 'penalty_std': False,
                'cutoff': None,
                'exclude': ['intercept', 'beta_u', 'alpha', 'zi_b1', 'zi_c'],
                'surveillance_name': None,
                'urbanisation_name': 'urbanisation_pop_weighted_std'}
            
            model_name = build_model_name_mult_stat(settings['alpha_type'], settings['alpha_parameters'],
                            settings['intercept_type'], settings['intercept_parameters'],
                            settings['link'], settings['link_stat_name'], settings['link_type'],
                            settings['b1_type'], settings['b1_parameters'],
                            settings['c_type'], settings['c_parameters'],
                            settings['stat_names'], settings['num_knots'], settings['knot_type'], settings['degree'],
                            settings['spline_implementation'], settings['spline_type'], settings['spline_parameters'],
                            settings['penalty_order'], settings['penalty_type'], settings['penalty_parameters'], settings['penalty_std'],
                            settings['cutoff'], settings['beta_u_type'], settings['beta_u_parameters'],
                            exclude=settings['exclude'],
                            surveillance_name=settings['surveillance_name'],
                            urbanisation_name=settings['urbanisation_name'])
            model_dict[model_name] = settings
        
    # Create tasks list
    tasks = list(model_dict.items())
    random.seed(42)  # for reproducibility
    random.shuffle(tasks)
    tasks = tasks[0::]
    #random.shuffle(tasks)
    
    print(f"Fitting {len(tasks)} models in total...")
    print(f"Data: {data_name}")
    
    # Number of workers (adjust based on your server)
    # Each model uses n_chains, so N_WORKERS * n_chains = total cores used
    N_WORKERS = 9
    
    with Pool(N_WORKERS, initializer=init_worker) as p:
        results = p.map(worker, tasks)
    
    # Print summary
    print("\n" + "="*50)
    print("Fitting Summary:")
    print("="*50)
    for model_name, status in results:
        print(f"{model_name}: {status}")

    # Now compare models after all fitting is done
    print("\n" + "="*50)
    print("Model Comparison:")
    print("="*50)
    metric="loo"
    comparison_df = compare_models(outpath, data_name, task=fitting_task, metric=metric)
    
    # Save comparison results
    save_path = os.path.join(outpath, f'{data_name}[{fitting_task}]', f"model_comparison({metric}).csv")
    comparison_df.to_csv(save_path)
    print(f"\nComparison saved to: {save_path}")

    elpd_metrics = pd.read_csv(os.path.join(outpath, f'{data_name}[{fitting_task}]', f"_model_elpd_metrics.csv"))
    elpd_metrics = elpd_metrics.sort_values(by='loo', ascending=False).reset_index(drop=True)
    elpd_metrics.to_csv(os.path.join(outpath, f'{data_name}[{fitting_task}]', f"_model_elpd_metrics.csv"), index=False)