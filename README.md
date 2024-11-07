## Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning (ICML 2024)<br><sub>Official Pytorch Implementation</sub>


## Updates
- [11/2024] Release paper.

## Setup

```shell
conda create -n ant python=3.9
conda activate ant
# Install torch, torchvision (https://pytorch.org/get-started/locally/)
pip install -e .
```
Download the dataset and put it in the `dataset` folder and Download the ckpt and put it in `initial_checkpoint` folder. Then run the following command.

## Run


### (1) Train classifier model
```shell
python train_discriminator.py --cfg configs/base_cls.yaml --source_data_path path_to_source_data --target_data_path path_to_target_data --initial_checkpoint initial_checkpoint/imagenet.ckpt --opts model.classifier.initial_checkpoint path_to_path_to_pretrain_classifier_checkpoint
```

### (2) Train ANT model
```shell
python train.py --cfg configs/base.yaml --source_data_path path_to_source_data --target_data_path path_to_target_data --opts model.classifier.initial_checkpoint path_to_path_to_finetuned_classifier_checkpoint  model.ddpm.initial_checkpoint path_to_path_to_pretrain_ddpm_checkpoint
```

```bibtex
@inproceedings{wang2024bridging,
  title={Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning},
  author={Wang, Xiyu and Lin, Baijiong and Liu, Daochang and Chen, Ying-Cong and Xu, Chang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

## Acknowledgement

We would like to express our gratitude for the contributions of several previous works to the development of VGen. This includes, but is not limited to [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [OpenCLIP](https://github.com/mlfoundations/open_clip), [guided-diffusion](https://github.com/openai/guided-diffusion), and [DDPM](https://github.com/hojonathanho/diffusion). We are committed to building upon these foundations in a way that respects their original contributions.