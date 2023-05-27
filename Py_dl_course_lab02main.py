import argparse
import os
import random
import re
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# add tensorboard for visibility
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

"""
解析用户输入的参数（args）；
如果用户指定了随机种子，设置 PyTorch、CUDA 和随机数生成器的种子，并开启 CUDA 模式下的确定性计算（deterministic），提醒用户此操作可能会降低训练速度，重新启动时可能出现意外行为；
如果用户指定了使用的 GPU，则禁用数据并行（data parallelism）；
如果用户将分布式连接地址（dist_url）设置为“env://”且未指定总的进程数量（world_size），从环境变量 WORLD_SIZE 中读取总的进程数量；
判断是否需要启用分布式训练（distributed），即是否存在多个 GPU 或多个节点，或者用户通过命令行参数明确要求启用多进程分布式训练；
计算每个节点可用的 GPU 数量（ngpus_per_node）；
根据是否需要启用分布式训练，调用不同的函数来进行训练，如果启用了分布式训练，则使用 torch.multiprocessing.spawn 函数在多个进程上运行 main_worker 函数。
"""
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

"""
设置当前进程（节点）的 GPU 编号和总的可用 GPU 数量；
如果需要启用分布式训练（distributed=True），则设置分布式训练相关参数，包括使用哪种后端（backend）、分布式连接地址（dist_url）、总的进程数量（world_size）、当前进程编号（rank）等；
根据命令行参数指定的模型（args.arch）创建模型，如果需要使用预训练模型（pretrained=True），则加载相应模型参数；并将全连接层的输出维度改为 args.out_features；
如果使用 GPU 训练，则将模型移动至 GPU 上，并根据是否启用分布式训练使用 torch.nn.parallel.DistributedDataParallel 或 torch.nn.DataParallel 进行数据并行处理（DataParallel 仅在单进程下使用，DistributedDataParallel 适用于多进程分布式训练）；
定义损失函数（CrossEntropyLoss）、优化器（SGD，包括学习率和动量）以及学习率调度器（StepLR）。
"""
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    # ----------change: use function 'torch.nn.Linear()' to change the output dimension of the fully connected layer------------------------#
    out_features = 200                      # output features for tiny-imagenet
    in_features = model.fc.in_features      # the original features of resnet
    model.fc = torch.nn.Linear(in_features, out_features)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
"""
如果 args.resume=True，则说明需要从之前的训练状态中恢复，此时先判断是否存在预训练的 checkpoint 文件，如果存在则进行下一步操作；
如果当前进程不在 GPU 上运行（即 args.gpu=None），则直接使用 torch.load 加载 checkpoint；
如果当前进程在 GPU 上运行（即 args.gpu 不为 None），则需要将 checkpoint 中的模型映射到指定的 GPU 上进行加载；
根据加载的 checkpoint，更新当前的 epoch、最佳准确度（best_acc1）、模型参数、优化器和学习率调度器等状态；
设置 cudnn.benchmark=True，以启用平台优化。
"""
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
"""
首先，设置 traindir 变量为 Tiny-ImageNet 训练集文件夹的路径；
然后，通过打开 val_annotations.txt 文件并读取其中的标签信息，将每个图像与其对应的标签建立映射关系，并存储在字典 d 中；
接下来，根据字典 d 的内容，为每个类别创建一个新的文件夹，用于存储 Tiny-ImageNet 验证集中该类别的所有图像；
对于每张图像，从原始 Tiny-ImageNet 验证集中读取其图像数据，并将其复制到相应的新文件夹中；
最后，设置 valdir 变量为 Tiny-ImageNet 新建的验证集文件夹的路径，并进行数据归一化处理。
"""
    # Data loading code
    # -------------- Change: We need to change the construction in valdir when using tiny-imagenet -------------#
    traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    f = open("/data/bitahub/Tiny-ImageNet/val/val_annotations.txt","r")       # the relative path of the file is corresponding to local path
    val_labels = f.readlines()      # get labels in val_annotations.txt to correct the original labels
    f.close()

    d = {}        # create a new dictionary to store the map between images and labels
    for item in val_labels:
        # split according to the format of val_annotations.txt
        image_name = item.split("\t")[0]
        image_label = item.split("\t")[1]
        if image_label not in d:
            d[image_label] = [image_name]
        else:
            d[image_label] += [image_name]
    for item in d:
        if not os.path.exists("/data/bitahub/Tiny-ImageNet/my_val/{}/images".format(item)):
            os.makedirs("/data/bitahub/Tiny-ImageNet/my_val/{}/images".format(item))
            # os.makedirs() could only been used when the path is not existed
        for num,img in enumerate(d[item]):
            source = "/data/bitahub/Tiny-ImageNet/val/images/{}".format(img)
            destination = "/data/bitahub/Tiny-ImageNet/my_val/{}/images/{}_{}.JPEG".format(item, item, num)
            # using shutil.copyfile() to create new val_dir just like the train_dir
            shutil.copyfile(source, destination)
    
    # the dataset on bitahub is "read-only" mode, so cann't write the new folder "my_val"
    for item in d:
        if not os.path.exists("/mydata/my_val/{}/images".format(item)):
            os.makedirs("/mydata/my_val/{}/images".format(item))
            # os.makedirs() could only been used when the path is not existed
        for num,img in enumerate(d[item]):
            source = "/data/bitahub/Tiny-ImageNet/val/images/{}".format(img)
            destination = "/mydata/my_val/{}/images/{}_{}.JPEG".format(item, item, num)
            # using shutil.copyfile() to create new val_dir just like the train_dir
            shutil.copyfile(source, destination)
    create new val_dir
    valdir = os.path.join(args.data, 'my_val')
    valdir = '/mydata/my_val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
"""
构建 Tiny-ImageNet 的训练集数据加载器
train_dataset：训练数据集，即 ImageFolder 对象；
batch_size：每个批次所包含的图像数目，即 args.batch_size；
shuffle：是否对数据进行打乱操作，若 train_sampler 为 None，则 shuffle = True；
num_workers：使用多少个子进程来加载数据，即 args.workers；
pin_memory：是否将数据存储于 CUDA 固定内存中，从而加速数据读取；
sampler：采样器，用于在分布式训练时对图像进行分片，并且保证每个 GPU 上的数据都来自不同的图片。
"""
    #--------------- Change: no need to reshape and flip in tiny-imagenet ---------------------#
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
"""
首先，代码使用 datetime.now().strftime() 函数获取当前时间，并且将其作为 tensorboard 存储的目录。之后，将训练数据加载器传递给 train() 函数，获得返回的 train_loss 和 train_acc 值。同时，将验证数据加载器传递给 validate() 函数，获得返回的 val_loss、acc1 和 val_acc 值。这些值将用于绘制 scalar（标量）图，以便能够更好地理解模型的训练和验证过程。
在每个 epoch 结束时，将会检查当前是否是最佳的模型，并在 args.save_path 路径下保存模型参数。在第 5 和 10 个 epoch 结束时，也会将模型参数保存在 args.save_path 路径下，以便查看模型在不同阶段的表现。
最后，在 TensorBoard 中绘制 scalar 图，记录训练 loss、训练 acc、验证 loss 和验证 acc 的变化情况。在训练过程结束后，关闭 TensorBoard 的写入器 writer，输出 "Training Finished"。
"""
    #--------------- Change: no need to reshape and flip in tiny-imagenet ---------------------#
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create Tensorboard and assign its storage dir as "datetime + name_of_web_construction"
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join('/output', 'logs', current_time + '_' + args.arch)
    writer = SummaryWriter(logdir)
    
    # Use Tendorboard to draw Graph of the net
    # dummy_input = torch.rand(4, 3, 64, 64)          # The corresponding input of tiny-imagenet
    # writer.add_graph(model, (dummy_input,))       # model in this .py file cannot be drawn by this way, because "torch.nn.DataParallel" is used when creating the model
    summary(model, (3, 64, 64))         # Instead, using lib "summary" to represent the construction

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        # -------------Change: We need more returned value to draw tensorboard--------------------#
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, val_loss, val_acc = validate(val_loader, model, criterion, args)
        
        scheduler.step()

        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
        # pick another two checkpoints for evaluation
        if epoch == 5 or epoch == 10:
             save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best,"checkpoint_epoch{}.pth.tar".format(epoch))

        # when one epoch finished, save scalar in tensorboard for visibility
        writer.add_scalar('scalar/train_loss', train_loss, epoch)
        writer.add_scalar('scalar/train_acc', train_acc, epoch)
        writer.add_scalar('scalar/val_loss', val_loss, epoch)
        writer.add_scalar('scalar/val_acc', val_acc, epoch)

    print("Training Finished")
    writer.close()

"""
在 train() 函数中，首先使用 AverageMeter 类初始化了 batch_time（每批数据的时间）、data_time（数据加载时间）、losses（损失）、top1（Top1 准确率）和 top5（Top5 准确率）五个度量器。之后，使用 ProgressMeter 类初始化了 progress，该类用于打印每个 epoch 的进度条。
在每个 epoch 开始时，将模型转换为训练模式 model.train()。对于每个 batch 数据，先记录数据加载所需时间 data_time，同时将数据 images 和标签 target 分别移动到 GPU 上（如果有）。
然后，计算模型的输出 output 和损失 loss，并通过 accuracy() 函数计算准确率 acc1 和 acc5。将损失值和准确率值记录在相应度量器中。
接着，清空优化器 optimizer 中的梯度信息 optimizer.zero_grad()，反向传播 loss.backward() 并更新模型参数 optimizer.step()。
最后，在每 args.print_freq（默认为 10） 个 batch 之后，打印当前进度 progress.display(i)。
最后，返回 loss 和 top5.avg（Top5 准确率的平均值）两个值。
"""
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    #---------------Change: Add more return values-----------------#
    return loss, top5.avg       # Our aim is to boost the average of {Top5 accuracy} of train_dataset to 95%
"""
在 validate() 函数中，依然是使用 AverageMeter 类分别初始化 batch_time（每批数据的时间）、losses（损失）、top1（Top1 准确率）和 top5（Top5 准确率）四个度量器。需要注意的是，在 top1 和 top5 的初始化中，我们使用了 Summary 枚举类来指定每次更新后是否影响其总数值，以及如何汇总每个 batch 中的值。
接着，在进入“测试模式” model.eval() 后，使用 with torch.no_grad() 包含 for 循环，避免计算图的构建和梯度更新，从而减少 GPU 存储需求和计算量。
在每个 batch 数据上，与 train() 函数类似，将数据 images 和标签 target 分别移动到 GPU 上，并计算模型的输出 output 和损失 loss。使用 accuracy() 函数计算准确率 acc1 和 acc5，同时在相应度量器中记录损失值和准确率值。
在计算完每个 batch 后，更新其他度量器 batch_time 和 progress。progress.display(i) 用于打印当前进度，如果 args.evaluate 为 True，则会额外打印出每张图片的预测结果和真实标签，最后调用 progress.display_summary() 打印所有 epoch 的总结信息。
最后，返回 top1.avg（Top1 准确率的平均值）、loss（损失值）和 top5.avg（Top5 准确率的平均值）三个值。
"""
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            if args.evaluate:       # when evaluating, show more detail about each picture
                print("pic:{:2}  aim:{:2}  result:{:4} {:>8}".format(i, int(target), int(output.argmax()), 'correct' if int(target) == int(output.argmax()) else 'false'))
                if i == 100:        # no need to caculate all 10,000 pics
                      break

        progress.display_summary()

    #---------------Change: Add more return values-----------------#
    return top1.avg, loss, top5.avg


#------------- no other changes in the following parts -------------------#
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()