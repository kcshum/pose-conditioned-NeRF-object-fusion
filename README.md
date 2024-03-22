# Language-driven Object Fusion into Neural Radiance Fields with Pose-Conditioned Dataset Updates (CVPR 2024)
Official Github repository for paper:
<p align="center">
  <a href="https://arxiv.org/abs/2309.11281"><i><b>Language-driven Object Fusion into Neural Radiance Fields with Pose-Conditioned Dataset Updates</b></i></a>
<br>
  <a href="https://scholar.google.com/citations?user=LAUhTjAAAAAJ"><i>Ka Chun Shum<sup>1</sup></i></a>, 
  <a href="https://ja-yeon-kim.github.io/"><i>Jaeyeon Kim<sup>1</sup></i></a>, 
  <a href="https://sonhua.github.io/"><i>Binh-Son Hua<sup>2</sup></i></a>, 
  <a href="https://ducthanhnguyen.weebly.com/"><i>Duc Thanh Nguyen<sup>3</sup></i></a>, 
  <a href="https://saikit.org/index.html"><i>Sai-Kit Yeung<sup>1</sup></i></a>
<br>
  <i><sup>1</sup>Hong Kong University of Science and Technology</i>&nbsp&nbsp <i><sup>2</sup>Trinity College Dublin</i>&nbsp&nbsp <i><sup>3</sup>Deakin University</i>
<br>
<br>
  <img width="900" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/52277184-576a-440a-a969-09597ead7b38">
<br>
</p>
We aim to insert an object into a NeRF background. 
We first customize and fine-tune a text-to-image diffusion model for view synthesis in an inpainting manner, then apply the model to progressively fuse an object into background views to update a background NeRF.



# Dataset
### 1. Data Management
Our dataset comprises 10 sets of multi-view object images and 8 sets of multi-view background images. 
All photos were taken with an iPhone and feature everyday backgrounds or objects.

Below is the visualization of some objects and backgrounds:
<p align="center">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/6569c6a5-24cb-4596-9cba-fd74823bcac4">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/cc8a975b-9fdc-4559-a8f8-5cae987b844c">&nbsp&nbsp
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/dac59422-47ed-40f8-8db6-674eb38ec4f6">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/a71b48d5-941b-407a-b00a-19c1062ac142">&nbsp&nbsp
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/813a9532-7df6-41d5-ae22-97940d26251a">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/b73f13c0-67f7-41b0-a498-98eff816016a">&nbsp&nbsp
<br>
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/78dbf441-5d44-4b87-aef2-eab232ab8910">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/131e70d6-93be-4b4e-ac5d-5e49f72c9efd">&nbsp&nbsp
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/81053344-3c98-487c-a6e6-bf7515bf6dae">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/240af5e0-4bfa-480a-8438-b0a54a77edf0">&nbsp&nbsp
  &nbsp&nbsp and more objects ...
</p>




<p align="center">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/04522694-3224-4350-8efd-98be0bcb8dec">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/7bb3dbfc-dab8-4b82-beb7-b81c16f36bff">&nbsp&nbsp
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/1a26815d-48e3-4bc9-a425-843a5a163951">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/aaaccf30-bb40-4b44-a639-cd614b92aa30">&nbsp&nbsp
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/b4d69ee6-e638-492e-b173-c162fee1426b">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/6938191c-c964-48d1-893a-f0b7cd29b846">&nbsp&nbsp
<br>
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/a5b31f8a-0736-4502-940a-d92daf6a27d4">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/1d10b6fa-d547-4804-8197-8b5ac13c1335">&nbsp&nbsp
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/5ecdfede-7387-4140-8e6d-4ed231d25e6f">
  <img width="112" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/06e6c3cc-b2d2-45eb-aef0-b6b7644c10f4">&nbsp
  &nbsp&nbsp and more backgrounds ...
</p>

### 2. Data Download
Our dataset can be downloaded [here](https://drive.google.com/file/d/1_Agyxvt8iAxKj8brDmqHYNAR7BHtD4ZL/view?usp=sharing). 
Unzip it and place the dataset folder as `pose-conditioned-NeRF-object-fusion/dataset`.

### 3. Customize your Data (Optional)
You may follow [instant-ngp](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) to construct your data.


# Environment
Clone the code and build a virtual environment for it:
```
git clone https://github.com/kcshum/pose-conditioned-NeRF-object-fusion.git
cd pose-conditioned-NeRF-object-fusion

conda create -n posefusion python=3.9
conda activate posefusion
```

We use [Paddle](https://github.com/PaddlePaddle/Paddle) implementation for diffusion model fine-tuning:
```
conda install -c conda-forge cudatoolkit=11.6 cudnn=8.4.1.50 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
pip install paddlepaddle-gpu==2.4.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddlenlp==2.5.2 ppdiffusers==0.14.0
```

>You are recommended to test whether Paddle is successfully installed by running in Python:
>```
>import paddle
>paddle.utils.run_check()
>```
>If cudnn cannot be detected, run the followings and try again:
>```
>conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/{change_to_your_dir}/anaconda3/envs/posefusion/include:/{change_to_your_dir}/anaconda3/envs/posefusion/lib
>conda deactivate
>conda activate posefusion
>```

We use Pytorch for NeRF optimization:
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install opencv-python kornia configargparse
```


# Training
### 1. Training Objective
We aim to insert an object *(represented by a set of multi-view object images)* into a background NeRF *(learned from a set of multi-view background images)*.

Below are the video visualization results of some edited NeRFs, where the red boxes refer to the target location:

<p align="center">
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/bfd0d8f2-8533-4e7c-90e6-f992f738ee7e">&nbsp
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/36b980e5-eb8c-49b1-b3aa-022ac7669e3d">&nbsp
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/7983a2ba-df1b-45a0-a545-1899f662112a">&nbsp
<br>
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/31ab6fbb-8277-4492-a022-dd22e312cc63">&nbsp
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/deffdb8c-e6bd-4539-96b7-a3d4352dd539">&nbsp
  &nbsp&nbsp and more edits ...
</p>

We provide in `configs/commands.txt` the configuration of various edits for you to try and modify. One example is described below.

### 2. Diffusion Model Fine-tuning for Object-blended View Synthesis
Fine-tune a diffusion model in inpainting manner on both the object and background images with customized text prompt:
```
python -u train_inpainting_dreambooth.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
--object_data="model_car" --background_data="wooden_table" \
--object_prompt="sks white model car" --background_prompt="pqp wooden table" \
--max_train_steps_OBJ=4000 --max_train_steps_BG=400
```
The command is self-explanatory. You may check the available arguments in the code.

The fine-tuned diffusion model is by default saved in `dream_outputs/{--object_data}_and_{--background_data}`.

### 3. NeRF Optimization with pose-conditioned dataset updates
Optimize a background NeRF and then insert the object:
```
python train_nerf_fusion.py \
--config configs/nerf_fusion.txt --datadir "dataset/background/wooden_table" \
--finetuned_model_path "dream_outputs/model_car_and_wooden_table" \
--prompt "sks white model car on pqp wooden table" \
--pivot_name "IMG_4853.png" --box_name wooden_table_02 \
--strength_lower_bound 35 --strength_higher_bound 35
```
The command is self-explanatory. You may check the available arguments in the code.

`--pivot_name` is the first view to train. `--box_name` is the bounding box to use.
`--strength_lower_bound` and `--strength_higher_bound` refer to the range of diffusion model noise strength used in inferencing for view refinement, 
here we keep it fixed as 35 (note: 0 = no noise; 100 = pure noise). You may try random noise by specifying `--strength_lower_bound 10 --strength_higher_bound 90`.

It is a prerequisite to provide a nice first object-blended view before updating the new nearby views. 
The updating dataset is visualized in `logs/{your_experiment}/visualization`.

The object bounding box is visualized in `logs/{your_experiment}/boundingbox`.

The NeRF renderings are periodically visualized in `logs/{your_experiment}/{epoch}_{training stage}`.

The NeRF model is periodically saved as `logs/{your_experiment}/{epoch}.tar`.


# Inferencing
After the training ends, you may reuse the previous command and specifying extra arguments as follows.

You may render all training views by running:
```
python train_nerf_fusion.py \
--config configs/nerf_fusion.txt --datadir "dataset/background/wooden_table" \
--finetuned_model_path "dream_outputs/model_car_and_wooden_table" \
--prompt "sks white model car on pqp wooden table" \
--pivot_name "IMG_4853.png" --box_name wooden_table_02 \
--strength_lower_bound 35 --strength_higher_bound 35 \
--render_image --ckpt_epoch_to_load 40000
```

You may render a video by running:
```
python train_nerf_fusion.py \
--config configs/nerf_fusion.txt --datadir "dataset/background/wooden_table" \
--finetuned_model_path "dream_outputs/model_car_and_wooden_table" \
--prompt "sks white model car on pqp wooden table" \
--pivot_name "IMG_4853.png" --box_name wooden_table_02 \
--strength_lower_bound 35 --strength_higher_bound 35 \
--render_video --ckpt_epoch_to_load 40000 --video_expname video_01 \
--video_frames 4842 4835 4854 4847 4871 4895 --num_Gaps 10
```

`--video_frames` defines a camera trajectory that goes through the specified views smoothly.
`--num_Gaps` is the number of interpolated novel views between each view.


# Acknowledgement
This code is built upon [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch) implementation of [instant-ngp](https://nvlabs.github.io/instant-ngp/), and [Paddle](https://github.com/PaddlePaddle/Paddle) implementation of [DreamBooth](https://dreambooth.github.io/). 
<br>
We thank them for their nice implementation!
