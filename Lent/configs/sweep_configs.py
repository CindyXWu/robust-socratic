# Uncomment and change values for sweep
from time import gmtime, strftime
import math

# Note I have an early stop criteria but this is optional
sweep_configuration = {
    'method': 'bayes',
    'name': strftime("%m-%d %H:%M:%S", gmtime()),
    'metric': {'goal': 'maximize', 'name': 'student test acc'},
    # CHANGE THESE
    'parameters': {
        #'spurious_corr': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}, # For grid search
        # 'alpha': {'distribution': 'uniform', 'min': 0, 'max': 1}, # For bayes search
        'lr': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},
        'final_lr': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
        'tau': {'distribution': 'log_uniform', 'min': -5, 'max': 2.3},
    },
    'early_terminate': {'type': 'hyperband', 'min_iter': 5}
}

t_sweep_configuration = {
    'method': 'bayes',
    'name': strftime("%m-%d %H:%M:%S", gmtime()),
    'metric': {'goal': 'maximize', 'name': 'teacher test acc'},
    # CHANGE THESE 
    'parameters': {
        'epochs': {'values': [1]},
        'lr': {'distribution': 'log_uniform', 'min': math.log(0.1), 'max': math.log(1)},
        'final_lr': {'distribution': 'log_uniform', 'min': math.log(0.05), 'max': math.log(0.1)}
    },
    # Iter refers to number of times in code the metric is logged
    'early_terminate': {'type': 'hyperband', 'min_iter': 5}
}
