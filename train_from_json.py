import os
import json
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Experiment import Experiment



parser = argparse.ArgumentParser(description='Lunch experiment from json file.')
parser.add_argument("-json", "--json", help="Json file.",
                    default='', type=str)
parser.add_argument("-epochs", "--epochs", help="Number of epochs.",
                    default=5000, type=int)
parser.add_argument("-resume", "--resume_from_checkpoint", help="Resume from checkpoint.",
                    default='', type=str)
parser.add_argument("-outdir", "--outdir", help="Output directory.",
                    default='', type=str)
args = parser.parse_args()

# define output directory
if not args.outdir == '':
    outdir = args.outdir
else:
    outdir = os.path.splitext(os.path.basename(args.json))[0]

# create it
os.makedirs(outdir, exist_ok=True)
print('Weights will be saved in {}'.format(outdir))

# read json file containing the configuration
with open(args.json) as f:
    conf = json.load(f)

# load model
model = Experiment(conf)


# callbacks = [EarlyStopping(monitor='r2/global', mode='max', patience=50)]
callbacks = [
      ModelCheckpoint(monitor='avg/r2/global', mode='max', save_top_k=1, save_last=True, filename='r2'),
      ModelCheckpoint(monitor='avg/mae/global', mode='min', save_top_k=1, save_last=True, filename='mae'),
      ModelCheckpoint(monitor='avg/mse/global', mode='min', save_top_k=1, save_last=True, filename='mse'),
      ModelCheckpoint(monitor='avg/rmse/global', mode='min', save_top_k=1, save_last=True, filename='rmse'),
      ModelCheckpoint(monitor='avg/pearson/global', mode='max', save_top_k=1, save_last=True, filename='pearson'),
]

# resume_from_checkpoint = 

trainer = pl.Trainer(gpus=1, max_epochs=args.epochs,
                    weights_save_path=outdir,
                    progress_bar_refresh_rate=20,
                    track_grad_norm = False,
                    auto_lr_find = True,
                    callbacks=callbacks)
# train
trainer.fit(model)
# load best
ckpt_path = trainer.checkpoint_callback.best_model_path
print(ckpt_path)
# test
modell = Experiment.load_from_checkpoint(ckpt_path)
trainer.test(modell, ckpt_path=ckpt_path)





# trainer = pl.Trainer(gpus=1, max_epochs=800,
#                     progress_bar_refresh_rate=20,
#                     track_grad_norm = False,
#                     auto_lr_find = True,
#                     resume_from_checkpoint = './lightning_logs/version_1/epoch=7999-step=7999.ckpt',
#                     callbacks=callbacks)
