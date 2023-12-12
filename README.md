# Inpainting Car Number Plate

## Prerequisites
This code has been tested on Ubuntu 14.04 and the following are the main components that need to be installed:
- Python3
- PyTorch 1.0+
- torchvision 0.2.0+
- tensorboardX
- pyyaml

## Train the model
```bash
python train.py --config configs/config.yaml
```

The checkpoints and logs will be saved to `checkpoints`。

You can set the path of train dataset(train_data_path) in `./configs/config.yaml`.
Currently, it is `./dataset/train`. This means that you need to save training dataset in the path.

## Test with the trained model and customised inpainting coordinates
By default, it will load the latest saved model in the checkpoints. You can also use `--iter` to choose the saved models by iteration.

Image input size should be [256,256].

```bash
python test_single_edit.py \
	--image examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png \
	--output examples/output.png \
	--x1 100 \
	--y1 50 \
	--x2 180 \
	--y2 210
```



## Inpainted examples

With PyTorch, the model was trained on ImageNet for 430k iterations to converge (with batch_size 48, about 150h). Here are some test results on the patches from ImageNet validation set.


| Input | Inpainted |
|:---:|:---:|
|<img width="250" alt="Screenshot 2023-12-10 at 8 56 41 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/b6a6e512-c6cd-471e-9139-4298b9320059">  | <img width="250" alt="Screenshot 2023-12-10 at 8 56 47 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/319b72f3-635e-433c-a278-496445c9c001">|
|<img width="250" alt="Screenshot 2023-12-10 at 8 57 14 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/41b1e048-0246-4c9d-b906-bdac7f3b0c96">|<img width="250" alt="Screenshot 2023-12-10 at 8 57 19 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/d89cbb8f-acd5-498f-9c5e-e431b98cce72">
|<img width="250" alt="Screenshot 2023-12-11 at 8 55 11 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/b0bb31c7-4bc7-4c8a-8205-0eadef7a8352">|<img width="250" alt="Screenshot 2023-12-11 at 8 55 17 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/2e0e2adc-10e5-4fa6-99e4-319ab5aa8065">|
