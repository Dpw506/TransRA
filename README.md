# TransRA
## TransRA: transformer and residual attention fusion for single remote sensing image dehazing(https://doi.org/10.1007/s11045-022-00835-x)
### Dependencies and Installation

* python3.8.5
* anaconda
* PyTorch =1.8.0
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)

### Pretrained Weights & Dataset

- Download [our model weights](https://drive.google.com/drive/folders/14waVgVo22oNGphUfE2yeXaNm6lJhGih_).
- Download Haze_1k [dataset](https://drive.google.com/drive/folders/1eeBA2V_l9-evSJ0XWhRAww6ftweq8hU_?usp=sharing)
- Download RICE [dataset](https://drive.google.com/drive/folders/1eeBA2V_l9-evSJ0XWhRAww6ftweq8hU_?usp=sharing)

### Dataset


#### Train
```shell
python train.py --data_dir data/Haze_1k/thick -train_batch_size 2 --model_save_dir train_result
```

#### Test
 ```shell
python test.py --model_save_dir results
 ```
 
 ## Qualitative Results

Quantitative comparisons over SateHaze1k for different methods:

<div style="text-align: center">
<img alt="" src="/images/thick.png" style="display: inline-block;" />
</div>

## Citation

If you use any part of this code, please kindly cite

```
@article{Dong2022,
  title={TransRA: transformer and residual attention fusion for single remote sensing image dehazing},
  author={Dong, Pengwei, Wang, Bo},
  journal={Multidimensional Systems and Signal Processing},
  url={https://doi.org/10.1007/s11045-022-00835-x},
  year={2022}
}
```
