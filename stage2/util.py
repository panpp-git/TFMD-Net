import os
import torch
import errno
import complexModules4

def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, args, epoch, module_type):
    checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(os.path.join(args.output_dir, module_type)):
        os.makedirs(os.path.join(args.output_dir, module_type))
    cp = os.path.join(args.output_dir, module_type, 'last.pth')
    fn = os.path.join(args.output_dir, module_type, 'epoch_'+str(epoch)+'.pth')
    torch.save(checkpoint, fn)
    symlink_force(fn, cp)

def load_freeze(fn, module_type, device = torch.device('cuda')):
    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    if module_type == 'layer1':
        args.device=device
        model = complexModules4.set_layer1_module(args)
    else:
        raise NotImplementedError('Module type not recognized')

    model.backbone.load_state_dict(checkpoint['model'])
    # 冻结backbone参数
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer, scheduler = set_optim(args, model, module_type)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_fr
    )
    return model, optimizer, scheduler, args, checkpoint['epoch']
def load(fn, module_type, device = torch.device('cuda')):
    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    if module_type == 'layer1':
        args.device=device
        model = complexModules4.set_layer1_module(args)
    else:
        raise NotImplementedError('Module type not recognized')
    model.load_state_dict(checkpoint['model'])
    optimizer, scheduler = set_optim(args, model, module_type)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint['epoch']


def set_optim(args, module, module_type):
    if module_type == 'layer1':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
        # optimizer=torch.optim.Adam(module.parameters(),lr=args.lr_fr)
    else:
        raise(ValueError('Expected module_type to be fr or fc but got {}'.format(module_type)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    return optimizer, scheduler


def print_args(logger, args):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    logger.info(message)

    args_path = os.path.join(args.output_dir, 'run.args')
    with open(args_path, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
