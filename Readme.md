# Efficient On-device Training via Gradient Filtering

> Yuedong Yang, Guihong Li, Radu Marculescu

This is the official repo for the paper `Efficient On-device Training via Gradient Filtering` accepted in CVPR 2023.

[arxiv](https://arxiv.org/abs/2301.00330) [video](https://www.youtube.com/watch?v=UGcKdzeTAnk)

<details><summary>Abstract</summary>

Despite its importance for federated learning, continuous learning and many other applications,
on-device training remains an open problem for EdgeAI.
The problem stems from the large number of operations (*e.g.*, floating point multiplications and additions) and memory consumption required during training by the back-propagation algorithm.
Consequently, in this paper, we propose a new gradient filtering approach which enables on-device CNN model training. More precisely, our approach creates a special structure with fewer unique elements in the gradient map, thus significantly reducing the computational complexity and memory consumption of back propagation during training.
Extensive experiments on image classification and semantic segmentation with multiple CNN models (*e.g.*, MobileNet, DeepLabV3, UPerNet) and devices (*e.g.*, Raspberry Pi and Jetson Nano) demonstrate the effectiveness and wide applicability of our approach. For example, compared to SOTA, we achieve up to 19 $\times$ speedup and 77.1\% memory savings on ImageNet classification with only 0.1\% accuracy loss. Finally, our method is easy to implement and deploy; over 20 $\times$ speedup and 90\% energy savings have been observed compared to highly optimized baselines in MKLDNN and CUDNN on NVIDIA Jetson Nano. Consequently, our approach opens up a new direction of research with a huge potential for on-device training.
</details>

## Features

### Reduce Computation and Memory Complexity for Backpropagation via Gradient Filter

<p align="center">
  <img src="assets/gf_method.png" />
</p>

Because of the high computation and memory complexity, backpropagation (BP) is the key bottleneck for CNN training. Our method reduces the complexity by introducing the gradient filter (highlighted in red in the bottom figure). The gradient filter approximates the gradient map with one consisting fewer unique elements and special structures. By doing so, operations in BP for a convolution layer can be greatly simplified, thus saving computation and memory.

### Over 10 $\times$ Speedup with Marginal Accuracy Loss 

<p align="center">
  <img src="assets/latency.png" />
</p>

Our method achieves significant speedup on both edge devices (Raspberry Pi 3 and NVIDIA Jetson Nano) and desktop devices with marginal accuracy loss.

## Environment Setup

1. Create and activate conda virtual environment
    ```
    conda create -n gradfilt python=3.8
    conda activate gradfilt
    ```

2. Install PyTorch 1.13.1
    
    Here we consider a system with x86_64 CPU, Nvidia GPU with CUDA 11.7, Ubuntu 20.04 OS. For systems with different configurations, please refer to pytorch's official [installation guide](https://pytorch.org/get-started/previous-versions/).
    ```
    conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

3. Install dependencies for the classification task

    ```
    pip install "jsonargparse[signatures]" pytorch_lightning==1.6.5 torchmetrics==0.9.2 pretrainedmodels
    git clone https://github.com/mit-han-lab/mcunet.git
    cd mcunet
    git checkout be404ea0dbb7402783e1c825425ac257ed35c5fc
    python setup.py install
    cd ..
    ```

4. Get checkpoint files
   Download from `https://drive.google.com/drive/folders/1OtohiuFwnRqcs82D9YsxjfS1n5SKoRKY?usp=sharing` and paste into classification folder

5. Runs the experiments
   The bash files are located in script_parallel folder. For example, to runs an experiment: use `CUDA_VISIBLE_DEVICES=x bash script_parallel/HOSVD_with_var_compression/mcunet/c10/c10_mcunet_var0.8.sh` => mcunet with cifar10 using HOSVD with variance is 0.8

Note: If you are missing the `six` module, please install it.
