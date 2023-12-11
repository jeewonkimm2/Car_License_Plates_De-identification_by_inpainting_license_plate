import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks import Generator
from utils.tools import get_config, is_image_file, default_loader, normalize, get_model_list

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--image', type=str)
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--x1', type=int, help='Top-left x-coordinate of the rectangular region')
parser.add_argument('--y1', type=int, help='Top-left y-coordinate of the rectangular region')
parser.add_argument('--x2', type=int, help='Bottom-right x-coordinate of the rectangular region')
parser.add_argument('--y2', type=int, help='Bottom-right y-coordinate of the rectangular region')


# Define the bbox2mask function for customised bbox according to coordinates
def bbox2mask(bbox, max_delta_h, max_delta_w, h, w):
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    y1, x1, y2, x2 = bbox
    y1 = y1
    x1 = x1
    y2 = y2
    x2 = x2
    mask[:, y1:y2, x1:x2] = 1.0
    return mask


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # Extract max_delta_h and max_delta_w from max_delta_shape in config
    max_delta_h, max_delta_w = config.get('max_delta_shape', [32, 32])

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:
        with torch.no_grad():
            if is_image_file(args.image):
                # Test a single ground-truth image with a mask at the specified rectangular region
                ground_truth = default_loader(args.image)
                ground_truth = transforms.ToTensor()(ground_truth)
                ground_truth = normalize(ground_truth)
                ground_truth = ground_truth.unsqueeze(dim=0)

                # Create a mask for the specified rectangular region
                mask = bbox2mask((args.y1, args.x1, args.y2, args.x2), max_delta_h, max_delta_w,
                                 config['image_shape'][1], config['image_shape'][0])
                mask = mask.unsqueeze(dim=0)

                # Set checkpoint path
                if not args.checkpoint_path:
                    checkpoint_path = os.path.join('checkpoints', config['dataset_name'],
                                                   config['mask_type'] + '_' + config['expname'])
                else:
                    checkpoint_path = args.checkpoint_path

                # Define the trainer
                netG = Generator(config['netG'], cuda, device_ids)
                # Latest model
                # last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
                last_model_name = get_model_list(checkpoint_path, "gen_00430000.pt")

                netG.load_state_dict(torch.load(last_model_name))
                model_iteration = args.iter
                print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))
                
                if cuda:
                    netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                    ground_truth = ground_truth.cuda()
                    mask = mask.cuda()

                # Inference
                x1, x2, offset_flow = netG(ground_truth, mask)
                inpainted_result = x2 * mask + ground_truth * (1. - mask)

                vutils.save_image(inpainted_result, args.output, padding=0, normalize=True)
                print("Saved the inpainted result to {}".format(args.output))
            else:
                raise TypeError("{} is not an image file.".format(args.image))
    except Exception as e:
        print("Error: {}".format(e))
        raise e

if __name__ == '__main__':
    main()
