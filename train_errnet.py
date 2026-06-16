from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'


def parse_pair(value, name):
    parts = [float(item.strip()) for item in value.split(',') if item.strip()]
    if len(parts) != 2:
        raise ValueError('{} must contain exactly two comma-separated values'.format(name))
    total = sum(parts)
    if total <= 0:
        raise ValueError('{} must have a positive sum'.format(name))
    return [item / total for item in parts]

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True, 
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

initial_fusion_ratio = parse_pair(opt.initial_fusion_ratio, 'initial_fusion_ratio')
finetune_fusion_ratio = parse_pair(opt.finetune_fusion_ratio, 'finetune_fusion_ratio')

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], initial_fusion_ratio)

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))

eval_dataset_real = datasets.CEILTestDataset(
    join(datadir, 'real20'),
    size=20,
    max_long_edge=512)

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)


"""Main Loop"""
engine = Engine(opt)

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

def joint_validation_score(ceil_meters, real_meters):
    metric = opt.best_metric
    total_weight = opt.joint_ceilnet_weight + opt.joint_real20_weight
    if total_weight <= 0:
        raise ValueError('joint validation weights must have a positive sum')

    ceil_weight = opt.joint_ceilnet_weight / total_weight
    real_weight = opt.joint_real20_weight / total_weight
    score = ceil_weight * ceil_meters[metric] + real_weight * real_meters[metric]
    if metric == 'LMSE':
        score = -score
    return score

if opt.resume:
    res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

# define training strategy
engine.model.opt.lambda_gan = 0
set_learning_rate(opt.lr)
while engine.epoch < opt.nEpochs:
    if opt.gan_start_epoch >= 0 and engine.epoch == opt.gan_start_epoch:
        engine.model.opt.lambda_gan = opt.gan_weight
        print('[i] enable GAN loss with weight {}'.format(opt.gan_weight))
    if engine.epoch == 30:
        set_learning_rate(5e-5)
    if engine.epoch == 40:
        set_learning_rate(1e-5)
    if opt.finetune_epoch >= 0 and engine.epoch == opt.finetune_epoch:
        print('[i] switch to real-heavy fusion ratio {}'.format(finetune_fusion_ratio))
        train_dataset_fusion.fusion_ratios = finetune_fusion_ratio
        set_learning_rate(opt.finetune_lr)
    if engine.epoch == 50:
        set_learning_rate(1e-5)

    engine.train(train_dataloader_fusion)
    
    if opt.eval_freq > 0 and engine.epoch % opt.eval_freq == 0:
        use_joint_best = opt.best_dataset == 'joint'
        ceil_loss_key = opt.best_metric if opt.best_dataset == 'ceilnet_table2' else None
        real_loss_key = opt.best_metric if opt.best_dataset == 'real20' else None
        maximize = opt.best_metric in ['PSNR', 'SSIM', 'NCC']
        ceil_meters = engine.eval(
            eval_dataloader_ceilnet,
            dataset_name='testdata_table2',
            loss_key=ceil_loss_key,
            maximize=maximize)
        real_meters = engine.eval(
            eval_dataloader_real,
            dataset_name='testdata_real20',
            loss_key=real_loss_key,
            maximize=maximize)
        if use_joint_best:
            score = joint_validation_score(ceil_meters, real_meters)
            engine.save_best_score(
                score,
                label='best_joint_{}'.format(opt.best_metric),
                message='joint_{} score'.format(opt.best_metric))
