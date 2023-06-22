
from nni.experiment import Experiment
from nni.algorithms.hpo.darts_tuner import DartsTuner

search_space = {} # ここにはDARTSの探索空間を指定します

tuner = DartsTuner() 

exp = Experiment(tuner, 'local')
exp.config.experiment_name = 'pytorch_darts'
exp.config.trial_concurrency = 1
exp.config.max_trial_number = 100
exp.config.search_space = search_space
exp.config.trial_command = '/usr/bin/python darts_sample.py'
exp.config.trial_code_directory = './'
exp.config.training_service.use_active_gpu = True
exp.config.max_trial_number_per_gpu = 1

exp.run()