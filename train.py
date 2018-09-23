import argparse
import importlib
import os
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("forkserver")

    import json
    from utils.utils import setup, get_optimizer, get_args_and_modules
    from utils.io_utils import save_yaml
    from exp_logger import setup_logging

    from torch.optim.lr_scheduler import ReduceLROnPlateau


    # Define main args
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument

    parser.add('--extension',  type=str, default="")
    parser.add('--config_name', type=str, default="")

    parser.add('--model', type=str, default="", help='')
    parser.add('--dataloader', type=str, default="", help='')
    parser.add('--runner', type=str, default="", help='')

    parser.add('--save_frequency',  type=int, default=1, help='')

    parser.add('--random_seed',     type=int, default=123, help='')
    parser.add('--experiments_dir', type=str, default="experiments", help='')
    parser.add('--comment', type=str, default='', help='Just any type of comment')

    parser.add('--optimizer', type=str, default='SGD', help='Just any type of comment')
    parser.add('--optimizer_args', default="lr=3e-3^momentum=0.9", type=str, help='separated with "^" list of args i.e. "lr=1e-3^betas=(0.5,0.9)"')

    parser.add('--num_epochs', type=int, default=200)

    parser.add('--patience',         type=int, default=5)
    parser.add('--lr_reduce_factor', type=float, default=0.3)

    parser.add('--no-logging', default=False, action="store_true")
    parser.add('--args-to-ignore', type=str, default="splits_dir, experiments_dir")

    parser.add('--set_eval_mode', action='store_true', default=False)
    parser.add('--device', type=str, default='cuda')


    # Gather args across modules
    args, default_args, m = get_args_and_modules(parser)

    # Setup logging and save dir
    args.save_dir = 'data' if args.no_logging else setup_logging(args, default_args, args.args_to_ignore.split(','))
    os.makedirs(f'{args.save_dir}/checkpoints', exist_ok=True)

    # Dump args
    save_yaml(vars(args), f'{args.save_dir}/args.yaml')
    # with open(f'{args.save_dir}/args.json', 'w') as f:
    #     json.dump(vars(args), f, indent=4)

    # Setup everything else
    setup(args)

    # Load splits and preprocess target
    model_native_transform = m['model'].get_native_transform()
    dataloader_train       = m['dataloader'].get_dataloader(args, model_native_transform, 'train')
    dataloader_val         = m['dataloader'].get_dataloader(args, model_native_transform, 'val')

    # Load model 
    model, criterion = m['model'].get_net(args, dataloader_train)

    # Load optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.lr_reduce_factor, verbose=True)

    for epoch in range(0, args.num_epochs):
        if args.set_eval_mode:
            model.eval()
        else:
            model.train()

        # Train
        torch.set_grad_enabled(True)
        m['runner'].run_epoch_train(dataloader_train, model, criterion, optimizer, epoch, args)
        
        # Validate
        model.eval()
        torch.set_grad_enabled(False)
        val_loss = m['runner'].run_epoch_test (dataloader_val,   model, criterion, epoch, args)
        
        scheduler.step(val_loss)

        # Save
        if (epoch != 0) and (epoch % args.save_frequency == 0):
            torch.save(model.state_dict(), f'{args.save_dir}/checkpoints/model_{epoch}.pth', pickle_protocol=-1)

