# Generative-inpainting-pytorch
A PyTorch reimplementation for the paper [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892) according to the author's [TensorFlow implementation](https://github.com/JiahuiYu/generative_inpainting).

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

Trained PyTorch model: [[Google Drive](https://drive.google.com/open?id=1qbfA5BP9yzdTFFmiOTvYARUYgW1zwBBK)] [[Baidu Wangpan](https://pan.baidu.com/s/17HzpiqMPLIznvCWBfpNVGw)]

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



## Test results on ImageNet validation set patches

With PyTorch, the model was trained on ImageNet for 430k iterations to converge (with batch_size 48, about 150h). Here are some test results on the patches from ImageNet validation set.

| Input | Inpainted |
|:---:|:---:|
| [![val_00000827_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_input.png)  | [![val_00000827_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00000827_output.png) |
| [![val_00008210_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png)  | [![val_00008210_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_output.png) |
| [![val_00022355_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_input.png)  | [![val_00022355_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00022355_output.png) |
| [![val_00025892_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_input.png)  | [![val_00025892_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00025892_output.png) |
| [![val_00045643_input](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_input.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_input.png)  | [![val_00045643_output](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_output.png)](examples/imagenet/imagenet_patches_ILSVRC2012_val_00045643_output.png) |
