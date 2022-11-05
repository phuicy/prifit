"""
Author: AruniRC
Date: Feb 2019
"""
import time
from src.utils import visualize_point_cloud_from_labels, visualize_point_cloud
import ipdb
import os
import os.path as osp
import data_utils
from data_utils.ShapeNetDataLoader import PartNormalDataset, SelfSupPartNormalDataset, ACDSelfSupDataset
from tensorboard_logger import configure, log_value
import itertools
import torch
from torch import nn
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import sys, ipdb
from args_parser import parse_args
from testing import evaluation
from itertools import product

torch.backends.cudnn.enabled = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# ipdb.set_trace()


def main(args):
    metrics = {'best_acc': 0, 'best_epoch': 0,'best_class_avg_miou': 0, 'best_instance_avg_miou': 0, 'best_chamfer_loss': np.inf}

    def log_string(str):
        logger.info(str)
        print(str)

    '''CUDA ENV SETTINGS'''
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.cudnn_off:
        torch.backends.cudnn.enabled = False  # needed on gypsum!

    # --------------------------------------------------------------------------
    '''CREATE DIR'''
    # --------------------------------------------------------------------------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg_shapenet')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        # if args.k_shot > 0:
        dir_name = args.model + '_ShapeNet_' + \
                   '_k-%d_seed-%d_lr-%.6f_lr-step-%d_lr-decay-%.2f_wt-decay-%.6f_l2norm-%d' \
                   % (args.k_shot, args.seed, args.learning_rate,
                      args.step_size, args.lr_decay, args.decay_rate,
                      int(args.l2_norm))
        if args.selfsup:
            dir_name = dir_name + '_selfsup-%s_margin-%.2f_lambda-%.2f' \
                       % (args.ss_dataset, args.margin, args.lmbda)
        if args.anneal_lambda:
            dir_name = dir_name + '_anneal-lambda_step-%d_rate-%.2f' \
                       % (args.anneal_step, args.anneal_rate)

        if args.quantile or args.msc_iterations:
            dir_name = dir_name + '_quantile-{}_msc-its-{}_max-num-clusters-{}_alpha-{}_beta-{}'.format(args.quantile,
                                                                                       args.msc_iterations,
                                                                                       args.max_num_clusters, args.alpha, args.beta)

        experiment_dir = experiment_dir.joinpath(dir_name)
        # else:
        #     experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # --------------------------------------------------------------------------
    '''LOG'''
    # --------------------------------------------------------------------------
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS ...')
    log_string(args)
    configure(log_dir)                                             # tensorboard logdir

    # --------------------------------------------------------------------------
    '''DATA LOADERS'''
    # --------------------------------------------------------------------------
    root = 'ShapeSelfSup/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal'

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split=args.train_split, normal_channel=False, k_shot=args.k_shot)
    TEST_DATASET  = PartNormalDataset(root=root, npoints=args.npoint, split=args.eval_split, normal_channel=False)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))

    num_part = args.num_parts

    log_string('Use self-supervision - alternate batches')
    if not args.retain_overlaps:
        log_string('\tRemove overlaps between labeled and self-sup datasets')
        labeled_fns = list(itertools.chain(*TEST_DATASET.meta.values())) \
                        + list(itertools.chain(*TRAIN_DATASET.meta.values()))
    else:
        log_string('\tUse all files in self-sup dataset')
        labeled_fns = []
    if args.ss_dataset == 'dummy':
        log_string('Using "dummy" self-supervision dataset (rest of labeled ShapeNetSeg)')
        SELFSUP_DATASET = SelfSupPartNormalDataset(root=root, npoints=args.npoint,
                                                    split='train', normal_channel=args.normal,
                                                    k_shot=args.n_cls_selfsup, labeled_fns=labeled_fns)
    elif args.ss_dataset == 'acd':
        log_string('Using "ACD" self-supervision dataset (ShapeNet Seg)')
        ACD_ROOT = args.ss_path
        SELFSUP_DATASET = ACDSelfSupDataset(root=ACD_ROOT, npoints=args.npoint,
                                            normal_channel=args.normal,
                                            k_shot=args.n_cls_selfsup,
                                            exclude_fns=labeled_fns, prefetch=False)

    selfsupDataLoader = torch.utils.data.DataLoader(SELFSUP_DATASET,
                                                    batch_size=args.batch_size,
                                                    shuffle=True, num_workers=1)
    selfsupIterator = iter(selfsupDataLoader)

    '''MODEL LOADING'''

    MODEL = importlib.import_module(args.model)
    shutil.copy('ShapeSelfSup/models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('ShapeSelfSup/models/pointnet_util.py', str(experiment_dir))
    if args.reconstruct:
        log_string('Reconstruction................................')
    if args.extra_layers:
        log_string('Extra Layers..................................')
    if 'dgcnn' in args.model:
        print('DGCNN params')
        classifier = MODEL.get_model(num_part, normal_channel=args.normal, k=args.dgcnn_k).cuda()
    else:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal, reconstruct=args.reconstruct, extra_layers=args.extra_layers, num_charts=args.num_charts, num_points=args.num_points).cuda()


    selfsupCriterion = MODEL.get_selfsup_loss(margin=args.margin).cuda()
    log_string("The number of self-sup data is: %d" % len(SELFSUP_DATASET))


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    if torch.cuda.device_count() > 1:
        log_string("Let's use {} GPUs!".format(torch.cuda.device_count()))
        classifier = nn.DataParallel(classifier)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
            )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    if args.pretrained_model is None:
        # Default: load saved checkpoint from experiment_dir or start from scratch
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            log_string('Use pretrained model from checkpoints')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier = classifier.apply(weights_init)
    else:
        # Path to a pre-trained model is provided (self-sup)
        log_string('Loading pretrained model from %s' % args.pretrained_model)
        start_epoch = 0
        ckpt = torch.load(args.pretrained_model)
        classifier.load_state_dict(ckpt['model_state_dict'])
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    #if torch.cuda.device_count() > 1:
    #    log_string("Let's use {} GPUs!".format(torch.cuda.device_count()))
    #    classifier = nn.DataParallel(classifier)
    if args.category:
        log_string("Using one hot")

    # --------------------------------------------------------------------------
    ''' MODEL TRAINING '''
    # --------------------------------------------------------------------------

    global_epoch = 0

    print('Using Ellipsoid as Primitive...............')

    if args.include_convex_loss:
        print('Using Convex Fitting/Convex Loss with lambda - {}.........................'.format(args.lmbda))
    if args.include_intersect_loss:
        print('Using Intersection Loss with alpha - {}..................................'.format(args.alpha))
        if args.include_pruning:
            print('Pruning Ellipsoids...................................................')
    if args.include_entropy_loss:
        print('Using Entropy Loss with beta - {}........................................'.format(args.beta))
    print('Using Categorical Cross Entropy Loss for Semantic Segmentation')

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        ''' Adjust (anneal) self-sup lambda '''
        if args.anneal_lambda:
            lmbda = args.lmbda * (args.anneal_rate ** (epoch // args.anneal_step))
        else:
            lmbda = args.lmbda

        '''learning one epoch'''
        num_iters = len(selfsupDataLoader)  # calc an epoch based on self-sup dataset
        for i in tqdm(list(range(num_iters)), total=num_iters, smoothing=0.9, desc='Training'):
            # ------------------------------------------------------------------
            #   SELF-SUPERVISED LOSS
            # ------------------------------------------------------------------
            try:
                data_ss = next(selfsupIterator)
            except StopIteration:
                # reached end of this dataloader
                selfsupIterator = iter(selfsupDataLoader)
                data_ss = next(selfsupIterator)

            points, chamfer_points, label, target = data_ss
            #points, label, target = data_ss
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            ############################CHAMFER POINTS###############################################
            chamfer_points = chamfer_points.data.numpy()
            chamfer_points[:, :, 0:3] = provider.random_scale_point_cloud(chamfer_points[:, :, 0:3])
            chamfer_points[:, :, 0:3] = provider.shift_point_cloud(chamfer_points[:, :, 0:3])
            chamfer_points = torch.Tensor(chamfer_points)
            ############################CHAMFER POINTS###############################################
            points, chamfer_points = points.float().cuda(), chamfer_points.float().cuda()

            points = points.transpose(2, 1)
            chamfer_points = chamfer_points.transpose(2, 1)
            # for self-sup category label is always unknown, so always zeros:

            optimizer.zero_grad()
            classifier.train()

            '''applying self-supervised Ellipsoid Fitting (Convex) loss'''
            #quantile_list = [0.02, 0.03, 0.05]
            points = chamfer_points[:, :, np.random.choice(5000, 2048, replace=False)]
            #print(chamfer_points.shape)
            #ipdb.set_trace()
            feat, loss_self_sup, chamfer_loss = classifier(points, chamfer_points=chamfer_points, include_convex_loss=args.include_convex_loss, include_intersect_loss=args.include_intersect_loss, include_entropy_loss=args.include_entropy_loss, include_pruning=args.include_pruning, quantile=args.quantile, msc_iterations=args.msc_iterations, max_num_clusters=args.max_num_clusters, alpha=args.alpha, beta=args.beta, batch_id=i, epoch=epoch, class_list=class_list, evaluation=False)
            ss_loss = torch.mean(loss_self_sup) * lmbda
            #print('loss: ', ss_loss.requires_grad)
            if i % 100 == 0:
                print('Final Loss: ', ss_loss.item())
                sys.stdout.flush()
            ss_loss.backward()
            optimizer.step()


        # ----------------------------------------------------------------------
        #   Logging metrics after one epoch
        # ----------------------------------------------------------------------
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Supervised loss is: %.5f' % loss_sup.data)
        # log_value('train_loss', loss_sup.data, epoch)

        if args.selfsup:
            log_string('Self-sup loss is: %.5f' % ss_loss.data)
            # log_value('selfsup_loss', ss_loss.data, epoch)

        # save every epoch
        savepath = str(checkpoints_dir) + ('/model_%03d.pth' % (epoch+1))
        log_string('Saving model at %s' % savepath)
        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        log_string('Saved model.')
        log_value('train_acc', train_instance_acc, epoch)
        log_value('train_lr', lr, epoch)
        log_value('train_bn_momentum', momentum, epoch)
        log_value('selfsup_lambda', lmbda, epoch)

        global_epoch += 1

    # ----------------------------------------------------------------------
    #   Evaluation on test-set after completing training epochs
    # ----------------------------------------------------------------------
        return_metrics = evaluation(args, epoch, classifier, metrics)


if __name__ == '__main__':

    args = parse_args()
    main(args)



