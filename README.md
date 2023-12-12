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

With PyTorch, the model was trained on our dataset. Here are some example results on the patches.


| Input | Inpainted |
|:---:|:---:|
|<img width="250" alt="Screenshot 2023-12-10 at 8 56 41 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/a282508c-aa39-4816-b756-dfb60c88987f">  | <img width="250" alt="Screenshot 2023-12-10 at 8 56 47 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/55e3d466-5dc0-43a1-801b-ee85d3a6fb1c">|
|<img width="250" alt="Screenshot 2023-12-10 at 8 57 14 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/fd666ad4-792b-4de9-b7c9-fad1d3ff0e15">|<img width="250" alt="Screenshot 2023-12-10 at 8 57 19 PM" src="https://github.com/jeewonkimm2/generative-inpainting-pytorch/assets/108987773/2ceccc88-7072-4bdc-a4fb-59ec0eec7fb5">

