import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.docker import DockerSyncer
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
        for cur_metric in ['mae','mse','rmse','r2']:
            metrics[cur_metric + '/' + cur_var] = 'avg/' + cur_metric + '/'+cur_var
    # define tune callback
    callbacks = [
                  TuneReportCallback(metrics, on="validation_end"),
                  ModelCheckpoint(monitor='avg/r2/global', save_top_k=1, save_last=True)
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
    ray.init(address='149.132.176.97:6379', _redis_password='5241590000000000')
    # define grid configuration
    config = {
        'train_csv': '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_train.csv',
        'val_csv': '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_val.csv',
        'test_csv': '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_val.csv',
        'src_prefix': 'spc.',
        'batch_size': 2000, # 10000
        'num_workers': 8,

        'powf': 4,
        'max_powf': 7,
        'insz': tune.grid_search([512, 1024, 2048]),
        'minsz': 4,
        'nsbr': 1,
        'leak': 0,
        'batch_momentum': 0.01,

        'learning_rate': tune.grid_search([0.001, 0.0001]),  #0.001,
        'weight_decay': 0.01,
        'loss': tune.grid_search(['l1','l2', 'classification']),
        'val': 'r2',
        'nbins': 10, # 10
        'tgt_vars': ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
    }

    # ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 180

    analysis = tune.run(
        train_grid_point,
        name = 'prova3',
        local_dir = '/home/flavio/ray_results',
        config = config,
        metric = 'r2/global',
        mode = 'max',
        resources_per_trial = {'cpu': 6, 'gpu': 1},
        resume = False
    )
    # resume = "ERRORED_ONLY"
            # local_dir = '/home/flavio/ray_results',
            # local_dir = '/home/ray/ray_results',
            # to sync with docker
            # sync_config = tune.SyncConfig(
            #     sync_to_driver=DockerSyncer
            # )
    try:
        best_config = analysis.get_best_config(metric="r2/global")
        # print("Best config: ", analysis.get_best_config(metric="global"))
        pprint(best_config)
        # Get a dataframe for analyzing trial results.
        df = analysis.dataframe()
        df.to_csv('grid_results_prova3.csv', sep=';', index=False)
    except:
        import ipdb; ipdb.set_trace()
