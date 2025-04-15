import argparse
import datetime
import logging
import math
import os
import random
import sys
import time
from os import path as osp
root_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录（basicsr）
sys.path.append(os.path.dirname(root_dir))  # 将根目录（NAFNet）加入路径
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # 分布式设置
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        init_dist(args.launcher, **opt['dist_params'] if 'dist_params' in opt else {})
    opt['rank'], opt['world_size'] = get_dist_info()

    # 随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # 设备设置
    opt['num_gpu'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    opt['device'] = 'cuda' if opt['num_gpu'] > 0 else 'cpu'

    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # 初始化 Wandb 和 TensorBoard 日志
    if opt['logger'].get('wandb') and opt['logger']['wandb'].get('project'):
        init_wandb_logger(opt)
    tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name'])) if opt['logger'].get('use_tb_logger') else None
    return logger, tb_logger

def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
    return train_loader, train_sampler, val_loader, total_epochs, total_iters

def main():
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    # 自动恢复训练状态
    state_folder_path = osp.join('experiments', opt['name'], 'training_states')
    resume_state = None
    if opt['rank'] == 0:
        try:
            states = os.listdir(state_folder_path)
            if states:
                max_state_file = f"{max(int(f.split('.')[0]) for f in states) if states else 0}.state"
                resume_state_path = osp.join(state_folder_path, max_state_file)
                if osp.exists(resume_state_path):
                    resume_state = resume_state_path
                    opt['path']['resume_state'] = resume_state
        except FileNotFoundError:
            pass

    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # 初始化目录和日志
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    logger, tb_logger = init_loggers(opt)

    # 创建数据加载器
    train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    # 初始化模型
    if resume_state:
        check_resume(opt,0) #check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: 0, iter: 0.") #logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        # start_epoch = resume_state['epoch']
        # current_iter = resume_state['iter']
        start_epoch = 0
        current_iter = 0
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # 初始化消息记录器
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 数据预取器配置
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
    if prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Using {prefetch_mode} prefetch dataloader')
        assert opt['datasets']['train'].get('pin_memory', True), "CUDA prefetch requires pin_memory=True"
    else:
        prefetcher = CPUPrefetcher(train_loader)

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    best_loss = float('inf')

    while current_iter <= total_iters:
        # 更新分布式采样器的 epoch
        if opt['dist']:
            train_sampler.set_epoch(start_epoch)

        prefetcher.reset()
        train_data = prefetcher.next()
        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break

            # 更新学习率
            model.update_learning_rate(current_iter, opt['train'].get('warmup_iter', -1))

            # 前向和反向传播
            model.feed_data(train_data)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            if result_code == -1:
                logger.error('Training stopped due to loss explosion.')
                exit(0)

            iter_time = time.time() - iter_time

            # 日志记录
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {
                    'epoch': start_epoch,
                    'iter': current_iter,
                    'total_iter': total_iters,
                    'time': iter_time,
                    'data_time': data_time,
                    'lrs': [model.get_current_learning_rate()]  # 确保是列表
                }
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

                # 添加检查和更新代码
                if 'lrs' not in log_vars:
                    log_vars['lrs'] = [model.get_current_learning_rate()]

                # 确保 log_vars['lrs'] 是列表
                if not isinstance(log_vars['lrs'], list):
                    log_vars['lrs'] = [log_vars['lrs']]

                # TensorBoard 记录
                if tb_logger:
                    for k, v in log_vars.items():
                        if k in ['l_total', 'l_pix', 'l_perceptual', 'l_gan']:
                            tb_logger.add_scalar(f'train/{k}', v, current_iter)
                    # 确保使用列表中的第一个元素
                    tb_logger.add_scalar('train/lr', log_vars['lrs'][0], current_iter)
                    tb_logger.flush()

            # 保存最佳模型
            if 'l_total' in model.log_dict and model.log_dict['l_total'] < best_loss:
                best_loss = model.log_dict['l_total']
                logger.info(f'New best model at iter {current_iter} (loss: {best_loss:.4f})')
                #model.save(start_epoch, current_iter, is_best=True)

            # 保存检查点
            # if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            #     logger.info('Saving models and training states.')
            #     model.save(start_epoch, current_iter)

            #     # 保存训练状态到文件
            #     if opt['rank'] == 0:
            #         save_states = {
            #             'iter': current_iter,
            #             'epoch': start_epoch,
            #             'state_dict': model.get_state_dict(),
            #             'optimizers': model.get_optimizers(),
            #             'schedulers': model.get_schedulers()
            #         }
            #         state_path = osp.join(state_folder_path, f"{current_iter}.state")
            #         torch.save(save_states, state_path)

            # 验证
            if opt.get('val') and current_iter % opt['val']['val_freq'] == 0:
                model.validation(val_loader, current_iter, tb_logger,
                                opt['val']['save_img'], opt['val'].get('rgb2bgr', True),
                                opt['val'].get('use_image', True))
                val_log = model.get_current_log()
                log_vars = {
                    'epoch': start_epoch,
                    'iter': current_iter,
                    'total_iter': total_iters,
                    'lrs': [model.get_current_learning_rate()]  # 确保是列表
                }
                log_vars.update(val_log)

                # 确保 log_vars['lrs'] 是列表
                if not isinstance(log_vars['lrs'], list):
                    log_vars['lrs'] = [log_vars['lrs']]

                msg_logger(log_vars)

                # TensorBoard 记录验证指标
                if tb_logger:
                    for k, v in val_log.items():
                        tb_logger.add_scalar(f'val/{k}', v, current_iter)
                    tb_logger.add_scalar('train/lr', log_vars['lrs'][0], current_iter)
                    tb_logger.flush()

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

        start_epoch += 1

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Saving the latest model.')
    #model.save(-1, -1)  # 保存最新模型

    # 最终验证
    if opt.get('val'):
        model.validation(val_loader, current_iter, tb_logger,
                        opt['val']['save_img'], opt['val'].get('rgb2bgr', True),
                        opt['val'].get('use_image', True))

    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main()