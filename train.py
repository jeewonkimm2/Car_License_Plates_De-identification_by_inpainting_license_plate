

import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from trainer import Trainer
import torchvision.transforms as transforms

from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, get_model_list, normalize
from utils.logger import get_logger
from model.networks import Generator
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

torch.autograd.set_detect_anomaly(True)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')



# Calculate performance metric of validation dataset
def validate(trainer, val_loader, config, iteration, writer, device):
    trainer.eval()
    total_loss_d = 0.0
    total_loss_g = 0.0  # Add this line
    total_loss_tv = 0.0

    iterable_val_loader = iter(val_loader)
    trainer_module = trainer.module
    start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1
    
    try:
        ground_truth = next(iterable_val_loader)
    except StopIteration:
        iterable_val_loader = iter(val_loader)
        ground_truth = next(iterable_val_loader)

    # Prepare the inputs
    bboxes = random_bbox(config, batch_size=ground_truth.size(0))
    x, mask = mask_image(ground_truth, bboxes, config)
    if device:
        x = x.cuda()
        mask = mask.cuda()
        ground_truth = ground_truth.cuda()
    
    # Perform inference
    losses, inpainted_result, _ = trainer(x, bboxes, mask, ground_truth)

    for k in losses.keys():
        if not losses[k].dim() == 0:
            losses[k] = torch.mean(losses[k])

    losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
    losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
            + losses['ae'] * config['ae_loss_alpha'] \
            + losses['wgan_g'] * config['gan_loss_alpha']

    # Calculate TV loss
    tv_loss = torch.sum(torch.abs(inpainted_result[:, :, :, :-1] - inpainted_result[:, :, :, 1:])) + \
            torch.sum(torch.abs(inpainted_result[:, :, :-1, :] - inpainted_result[:, :, 1:, :]))


    # Accumulate the validation loss
    total_loss_d += losses['d'].item()
    total_loss_g += losses['g'].item()  # Add this line
    total_loss_tv += tv_loss.item()  # Add this line


    # Calculate average validation loss
    avg_loss_d = total_loss_d / len(val_loader)
    avg_loss_g = total_loss_g / len(val_loader)  # Add this line
    avg_loss_tv = total_loss_tv / len(val_loader)

    # Print or log the average validation loss
    print(f'Average Validation Loss (Discriminator): {avg_loss_d}')
    print(f'Average Validation Loss (Generator): {avg_loss_g}')  # Add this line
    print(f'Average Validation Loss (TV): {avg_loss_tv}')



    writer.add_scalar('val_loss_d', avg_loss_d, iteration)
    writer.add_scalar('val_loss_g', avg_loss_g, iteration)  # Add this line

    return avg_loss_tv, avg_loss_d, avg_loss_g





def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # Store values for visualization
    avg_loss_tv_list = []  
    avg_loss_g_list = []
    avg_loss_d_list = []

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                config['dataset_name'],
                                config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(logdir=checkpoint_path)
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call

    logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info("Configuration: {}".format(config))




    try:  # for unexpected error logging
        # Load the dataset
        logger.info("Training on dataset: {}".format(config['dataset_name']))
        train_dataset = Dataset(data_path=config['train_data_path'],
                                with_subfolder=config['data_with_subfolder'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'])
        val_dataset = Dataset(data_path=config['val_data_path'],
                            with_subfolder=config['data_with_subfolder'],
                                image_shape=config['image_shape'],
                            random_crop=config['random_crop'])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                num_workers=config['num_workers'])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=False,
                                                num_workers=config['num_workers'])

        # Define the trainer
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))

        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        # Get the resume iteration to restart training
        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1

        iterable_train_loader = iter(train_loader)

        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):


            try:
                ground_truth = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth = next(iterable_train_loader)

            # Prepare the inputs
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            losses, inpainted_result, offset_flow = trainer(x, bboxes, mask, ground_truth, compute_g_loss)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            # trainer_module.optimizer_d.step()

            # Update G
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                            + losses['ae'] * config['ae_loss_alpha'] \
                            + losses['wgan_g'] * config['gan_loss_alpha']
                losses['g'].backward()


                trainer_module.optimizer_d.step()
                trainer_module.optimizer_g.step()

            # Log and visualization
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses.get(k, 0.)
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)
        

            if iteration % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                            offset_flow[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                nrow=3 * 4,
                                normalize=True)
                
            avg_loss_tv, avg_loss_d, avg_loss_g = validate(trainer, val_loader, config, iteration, writer, device=torch.device('cuda' if cuda else 'cpu'))
            avg_loss_tv_list.append(avg_loss_tv)  # append the value for visualization
            avg_loss_d_list.append(avg_loss_d)
            avg_loss_g_list.append(avg_loss_g)

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)


        # Save avg_loss_tv_list as a CSV file
        csv_file_path = ('./result/avg_loss_tv_list.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Avg Loss TV'])
            for i, avg_loss_tv in enumerate(avg_loss_tv_list):
                csv_writer.writerow([avg_loss_tv])



        # Save avg_loss_g_list as a CSV file
        csv_file_path = ('./result/avg_loss_g_list.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Avg Loss G'])
            for i, avg_loss_g in enumerate(avg_loss_g_list):
                csv_writer.writerow([avg_loss_g])

        # Save avg_loss_d_list as a CSV file
        csv_file_path = ('./result/avg_loss_d_list.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Avg Loss D'])
            for i, avg_loss_d in enumerate(avg_loss_d_list):
                csv_writer.writerow([avg_loss_d])

        # Visualization of avg_loss_g and avg_loss_d after training
        plt.plot(avg_loss_g_list, label='avg_loss_g')
        plt.plot(avg_loss_d_list, label='avg_loss_d')
        plt.title('Average Generator and Discriminator Loss of Validation Set')
        plt.xlabel('Iteration')
        plt.ylabel('Average Loss')
        plt.legend()

        # Save the plot as an image
        plt.savefig('./result/train_g_d_loss_plt')

        # Close the plot
        plt.close()

            
            



    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    main()
