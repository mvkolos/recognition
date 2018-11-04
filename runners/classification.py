import os.path
import time
import torch
import numpy as np
import tqdm
from runners.common import AverageMeter, accuracy, print_stat
from huepy import red, lightblue, orange, cyan
from utils.task_queue import TaskQueue


def get_args(parser):
    parser.add('--use_mixup',  default=False, action='store_true')
    parser.add('--mixup_alpha', type=float, default=0.1)

    parser.add('--print_frequency', type=int, default=50)
    
    return parser

def check_data(data): #dataloader):
   (names, x_, y_)  = data #iter(dataloader_train).next()[1].to(args.device)

   assert isinstance(names, list) and isinstance(x_, torch.cuda.FloatTensor) and isinstance(y_, list), red("expecting each output to be a list of tensors")
   assert len(names) == len(y_)
   assert len(x_) == 1 or len(x_) == len(y_)

data = dict(last_it=0)

def run_epoch(dataloader, model, criterion, optimizer, epoch, args, part='train'):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()
    
    writer = run_epoch.writer
    outputs = []

    tq = TaskQueue(maxsize=64, num_workers=5, verbosity=0)  

    end = time.time()
    
    for it, (names, x_, *y_) in enumerate(dataloader):
        

        # check_data((names, x_, y_))
        
        y = [y.to(args.device, non_blocking=True) for y in y_]
        x = x_.to(args.device, non_blocking=True)
        
        # Measure data loading time
        data_time.update(time.time() - end)

        output = model(x)
        
        loss = criterion(output, y) 
        
        if part=='train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loss_meter.update(loss.item())

        # For multitarget tasks we want to get acc per target
        acc_per_target = accuracy(output, y)
        acc_meter.update(np.mean([x for x in acc_per_target if x != -1]))


        if 'save_driver' in args and args.save_driver is not None:
            for num, (name, x) in enumerate(zip(names, x_)):
                y = [y[num].detach().cpu().numpy() for y in y_]
                pred = [o[num].detach().cpu().numpy() for o in output]

                tq.add_task(npz_per_item, {
                    # 'input':  _x.detach().cpu().numpy(),
                    # 'target': y,
                    'pred':   pred,#.detach().cpu().numpy(),
                    'name':   str(name),
                }, path=f'{args.dump_path}/{os.path.basename(name)}.png', args=args)  

        # Logging 
        loss_meter.update(loss.item(), x.shape[0])

        if part == 'train':
            writer.add_scalar(f'Metrics/{part}/loss', loss.item(),   data['last_it'])
            writer.add_scalar(f'Metrics/{part}/acc', acc_meter.val,  data['last_it'])

            # writer.add_scalars(f'metrics/{part}/accs_per_target', {str(k): acc_per_target[k] for k in range(len(acc_per_target))}, last_it[part])
            data['last_it'] += 1

        if it % args.print_frequency == 0:
            print(f'{lightblue(part.capitalize())}: [{epoch}][{it}/{len(dataloader)}]\t'\
                  f'{print_stat("Time", batch_time.val, batch_time.avg)}\t'\
                  f'{print_stat("Data", data_time.val, data_time.avg)}\t'\
                  f'{print_stat("Loss", loss_meter.val, loss_meter.avg, 4)}\t',
                  f'{print_stat("Acc",  acc_meter.val,  acc_meter.avg, 4)}\t')
        
    tq.stop_()
    # tq.join()
    print(f' * \n'
          f' * Epoch {epoch} {red(part.capitalize())}:\t'
          f'Loss {loss_meter.avg:.4f}\t'
          f'Acc  {acc_meter.avg:.4f}\t'
          f' *\t\n')

    if part != 'train':
        writer.add_scalar(f'Metrics/{part}/loss', loss_meter.avg,  data['last_it'])
        writer.add_scalar(f'Metrics/{part}/acc',  acc_meter.avg,   data['last_it'])

    return loss_meter.avg


def npz_per_item(data, path, args):
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys

    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    np.savez_compressed(path, **data)




