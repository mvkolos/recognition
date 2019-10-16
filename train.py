# torch.multiprocessing.set_start_method("forkserver")
from utils.exp_logger import setup_logging_from_config
from huepy import yellow 
# from models.model import save_model
from munch import munchify
from pathlib import Path

from utils.io_utils import save_yaml, load_yaml
from utils.eval import eval_modules, eval_paths
from utils.utils import setup

import argparse
import os
import sys
import time
import torch
from torch import nn
import torch.multiprocessing

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
RECOGNITION_PATH = os.path.abspath(FILE_PATH + '/..')

torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.yml')
    return parser

parser = get_args()


# Gather args across modules
args = parser.parse_args()

config = load_yaml(args.config)
extension = config['extension']
sys.path.append(RECOGNITION_PATH)
sys.path.append(f'extensions/{extension}')


# Setup logging and creates save dir
writer = setup_logging_from_config(config, exp_name_use_date=True)

# Dump args
save_yaml(config, Path(config['experiment_dir'])/'config.yml')

eval_modules(config)
eval_paths(config)

config['dumps_dir'] = config['experiment_dir']/'dumps'



# Setup everything else
# setup(config)

# Load dataloaders
data_factory = config['data_factory_fn'](config)
    
dataloader_train = data_factory.make_loader(is_train=True)
dataloader_val = data_factory.make_loader(is_train=False)

print(len(dataloader_train), len(dataloader_val))
wrapper = config['wrapper_fn']()
pipeline = wrapper.get_pipeline(config, data_factory)#.to(device)
     
train_factory = config['train_factory_fn']

# Load runner
runner = config['runner_fn']#(config)

# Load saver
# saver = get_saver('DummySaver')



# def set_param_grad(model, value, set_eval_mode=True):
#     for param in model.parameters():
#         param.requires_grad = value
    
#     if set_eval_mode:
#         model.eval()

# Run 
for stage_num, (stage_name, stage_args_) in enumerate(config['stages'].items()):

    print (yellow(f' - Starting stage "{stage_name}"!'))

    stage_args = munchify({**stage_args_, **config['train_args'] })    

    optimizers = train_factory.make_optimizers(stage_args, pipeline)
    schedulers = train_factory.make_schedulers(stage_args, optimizers, pipeline)
    
    
    print('schedulers', schedulers.keys())
    train_factory.set_trainable(stage_args, pipeline)
    criterions = train_factory.make_criterions(stage_args)
    
    print('criterions', criterions.keys())
    
    if stage_args.parallel:
        pipeline = nn.DataParallel(pipeline)
#         print(pipeline)

#     if args.fp16:
#         import apex 

#         optimizer = apex.fp16_utils.FP16_Optimizer(optimizer, dynamic_loss_scale=True, verbose=False)
#     else:
#         optimizer.backward = lambda x: x.backward()
        
    # Go
    for epoch in range(0, stage_args.num_epochs):
        dumps_dir = config['dumps_dir']/str(epoch)
        os.makedirs(dumps_dir, exist_ok=True)
        pipeline.train()
        
#         if epoch % stage_args.save_frequency == 0:
#             if isinstance(pipeline, nn.DataParallel):
#                 pipeline.module.save(config['dumps_dir'], epoch, stage_args)
#             else:
#                 pipeline.save(config['dumps_dir'], epoch, stage_args)
#             if stage_args.save_optimizers:
#                 torch.save(optimizers, config['dumps_dir']/f'optimizers{epoch}')
        
        # ===================
        #       Train
        # ===================
        with torch.set_grad_enabled(True):
            runner.run_epoch(dataloader_train, pipeline, criterions, optimizers, epoch, stage_args, phase='train', writer=writer, dumps_dir=dumps_dir)
        


        # ===================
        #       Validate
        # ===================
        torch.cuda.empty_cache()
        
        pipeline.eval()
        
        
        with torch.set_grad_enabled(False):
            if stage_args.supervised_eval:
                val_loss = runner.run_epoch(dataloader_val, pipeline, criterions, None, epoch, stage_args, phase='val', writer=writer, dumps_dir=dumps_dir)
            else:
                runner.evaluate(dataloader_val, pipeline, epoch, stage_args, writer)
            
        
        for k, scheduler in schedulers.items():
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and stage_args.supervised_eval:
                scheduler.step(val_loss)
            else:
                scheduler.step()


        # Save
        if epoch % stage_args.save_frequency == 0:
            if isinstance(pipeline, nn.DataParallel):
                pipeline.module.save(config['dumps_dir'], epoch, stage_args)
            else:
                pipeline.save(config['dumps_dir'], epoch, stage_args)
            if stage_args.save_optimizers:
                torch.save(optimizers, config['dumps_dir']/f'optimizers{epoch}')
#             save_model(model, epoch, stage_args, optimizer, stage_num)