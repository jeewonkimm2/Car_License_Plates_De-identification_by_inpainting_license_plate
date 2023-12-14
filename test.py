import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from utils.tools import get_config
from data.dataset import Dataset
from model.networks import Generator
from trainer import Trainer
from utils.tools import get_config, random_bbox, mask_image


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)


# Calculate PSNR for performance metric
def calculate_psnr(original, restored, max_value=1.0):
    mse = torch.mean((original - restored) ** 2)
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()

def validate_test(trainer, test_loader, config, iteration, writer, device):
    # Implement the validation logic for the test_loader
    # You can refer to the existing validate function and modify it as needed
    trainer.eval()
    total_loss_d = 0.0
    total_loss_g = 0.0  # Add this line
    total_loss_tv = 0.0
    total_psnr = 0.0

    iterable_val_loader = iter(test_loader)
    trainer_module = trainer.module
    start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1
    
    try:
        ground_truth = next(iterable_val_loader)
    except StopIteration:
        iterable_val_loader = iter(test_loader)
        ground_truth = next(iterable_val_loader)

    # Prepare the inputs
    bboxes = random_bbox(config, batch_size=ground_truth.size(0))
    x, mask = mask_image(ground_truth, bboxes, config)
    if device:
        x = x.cuda()
        mask = mask.cuda()
        ground_truth = ground_truth.cuda()
    
    bboxes = random_bbox(config, batch_size=x.size(0))

        # Inference
    x1, x2, offset_flow = trainer(ground_truth, mask)
    inpainted_result = x2 * mask + ground_truth * (1. - mask)

    # Calculate TV loss
    tv_loss = torch.sum(torch.abs(inpainted_result[:, :, :, :-1] - inpainted_result[:, :, :, 1:])) + \
            torch.sum(torch.abs(inpainted_result[:, :, :-1, :] - inpainted_result[:, :, 1:, :]))

    # Calculate PSNR of
    psnr_value = calculate_psnr(ground_truth, inpainted_result)

    total_loss_tv += tv_loss.item()
    total_psnr += psnr_value


    avg_loss_tv = total_loss_tv / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)

    print(f'Average Validation Loss (TV): {avg_loss_tv}')
    print(f'Average Validation PSNR: {avg_psnr}')

    return avg_loss_tv, avg_psnr




def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = torch.randint(1, 10000, (1,)).item()
    print("Random seed: {}".format(args.seed))
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:
        with torch.no_grad():

            # Define the generator
            netG = Generator(config['netG'], cuda, device_ids)

            # Load the model checkpoint
            checkpoint_file = os.path.join(args.checkpoint_path, f'gen_00060000.pt')
            if os.path.exists(checkpoint_file):
                # checkpoint = torch.load(checkpoint_file)
                netG.load_state_dict(torch.load(checkpoint_file))
                # netG.load_state_dict(checkpoint['state_dict'])
            else:
                raise FileNotFoundError(f"Checkpoint file '{checkpoint_file}' does not exist.")

            if cuda:
                netG = nn.parallel.DataParallel(netG, device_ids=device_ids)

            # Load the test dataset
            test_dataset = Dataset(data_path=config['test_data_path'],
                                   with_subfolder=config['data_with_subfolder'],
                                   image_shape=config['image_shape'],
                                   random_crop=config['random_crop'])
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=config['batch_size'],
                                                      shuffle=False,
                                                      num_workers=config['num_workers'])

            # Create SummaryWriter for tensorboard logging
            writer = SummaryWriter(logdir=args.checkpoint_path)

            # Perform test/validation
            validate_test(netG, test_loader, config, args.iter, writer, device=torch.device('cuda' if cuda else 'cpu'))

    except Exception as e:
        print("Error: {}".format(e))
        raise e

if __name__ == '__main__':
    main()
