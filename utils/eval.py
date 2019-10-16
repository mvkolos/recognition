import pydoc
from pathlib import Path


def eval_modules(config):
    for arg in config:
        if isinstance(arg, str) and arg.endswith('_fn'):
            if not config[arg]:
                config[arg] = None
                continue
            config[arg] = pydoc.locate(config[arg])
        elif isinstance(config[arg], dict):
            eval_modules(config[arg])
            
def eval_paths(config):
    for arg in config:
        if isinstance(arg, str) and arg.endswith('_dir'):
            if config[arg]:
                config[arg] = Path(config[arg])
            else:
                config[arg] = None
        elif isinstance(config[arg], dict):
            eval_paths(config[arg])