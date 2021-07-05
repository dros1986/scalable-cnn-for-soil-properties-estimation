import ray
import json
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.docker import DockerSyncer
from ray.tune.schedulers import ASHAScheduler
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from Experiment import Experiment
from pprint import pprint

def train_grid_point(config, data_dir=None, max_epochs=5000, num_gpus=1): # 5000 10000
    # define model
    model = Experiment(config)
    # define metrics
    metrics = {}
    for cur_var in config['tgt_vars'] + ['global']:
        for cur_metric in ['mae','mse','rmse','r2', 'pearson']:
            metrics[cur_metric + '/' + cur_var] = 'avg/' + cur_metric + '/'+cur_var
    # define tune callback
    callbacks = [
                  TuneReportCallback(metrics, on="validation_end"),
                  ModelCheckpoint(monitor='avg/r2/global', mode='max', save_top_k=1, save_last=True, filename='r2'),
                  ModelCheckpoint(monitor='avg/mae/global', mode='min', save_top_k=1, save_last=True, filename='mae'),
                  ModelCheckpoint(monitor='avg/mse/global', mode='min', save_top_k=1, save_last=True, filename='mse'),
                  ModelCheckpoint(monitor='avg/rmse/global', mode='min', save_top_k=1, save_last=True, filename='rmse'),
                  ModelCheckpoint(monitor='avg/pearson/global', mode='max', save_top_k=1, save_last=True, filename='pearson'),
                ]
    # define trainer
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=max_epochs, progress_bar_refresh_rate=20, callbacks=callbacks)
    trainer.fit(model)


# ASHAScheduler selects the most promising
# https://stackoverflow.com/questions/44260217/hyperparameter-optimization-for-pytorch-model

# /home/flavio/workspace/pignoletto/data
#


if __name__ == '__main__':
    # init ray
    # ray.init(address='localhost:6379', _redis_password='5241590000000000')
    ray.init(address='149.132.176.97:6379', _redis_password='5241590000000000')
    # define grid configuration
    config = {
        'train_csv': '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_train.csv',
        'val_csv': '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_val.csv',
        'test_csv': '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_test.csv',
        'src_prefix': 'spc.',
        'batch_size': 2000, # 10000
        'num_workers': 8,
        'fmin': tune.randint(400, 1200),
        'fmax': tune.randint(2300, 2500),
        # 'fmin': tune.choice([450, 800, 1200]),
        # 'fmax': tune.choice([2300, 2400, 2500]),

        'powf': 4,
        'max_powf': 7,
        'insz': tune.choice([512, 1024, 2048]),
        'minsz': 4,
        'nsbr': tune.randint(0, 3),
        'leak': tune.uniform(0, 0.2),
        # 'leak': tune.choice([0, 0.2]),
        'batch_momentum': 0.01,
        'use_batchnorm': tune.choice([True, False]),
        'use_gap': False,

        'learning_rate': tune.uniform(0.001, 0.0001),
        'weight_decay': tune.uniform(0, 0.05),
        'loss': tune.choice(['l1','l2', 'classification']),
        'val': 'r2',

        'nbins': 10, # 'nbins': tune.choice([10, 20, 30]),
        'tgt_vars': ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
    }

    # ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 180
    asha_scheduler = ASHAScheduler(
        time_attr='epoch',
        # metric='r2/global',
        # mode='max',
        max_t=1000,
        grace_period=100,
        reduction_factor=4,
        brackets=1
    )


    analysis = tune.run(
        train_grid_point,
        name = 'asha3',
        local_dir = '/home/flavio/ray_results',
        config = config,
        metric = 'r2/global',
        mode = 'max',
        resources_per_trial = {'cpu': 6, 'gpu': 1},
        # resources_per_trial = {'cpu': 4, 'gpu': 0.5},
        scheduler=asha_scheduler,
        resume = False
        # resume = "ERRORED_ONLY"
    )

    try:
        best_config = analysis.get_best_config(metric="r2/global")
        # print("Best config: ", analysis.get_best_config(metric="global"))
        pprint(best_config)
        # Get a dataframe for analyzing trial results.
        df = analysis.dataframe()
        df.to_csv('asha3.csv', sep=';', index=False)
        with open('asha3.json', 'w') as f:
            json.dump(best_config, f, ensure_ascii=False, indent=4)
    except:
        import ipdb; ipdb.set_trace()
