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
The data 

### 2. Data Download
> Todo: data coming soon!



# Training
### 1. Training Objective
We aim to insert an object *(represented by a set of multi-view object images)* into a background NeRF *(learned from a set of multi-view background images)*.

Below are the video visualization results of some edited NeRFs, where the red boxes refer to the target location:

<p align="center">
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/bfd0d8f2-8533-4e7c-90e6-f992f738ee7e">&nbsp
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/bfd0d8f2-8533-4e7c-90e6-f992f738ee7e">&nbsp
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/bfd0d8f2-8533-4e7c-90e6-f992f738ee7e">&nbsp
<br>
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/bfd0d8f2-8533-4e7c-90e6-f992f738ee7e">&nbsp
  <img width="256" src="https://github.com/kcshum/pose-conditioned-NeRF-object-fusion/assets/41816098/bfd0d8f2-8533-4e7c-90e6-f992f738ee7e">&nbsp
  &nbsp&nbsp and more ...
</p>


### 2. Diffusion Model Fine-tuning for Object-blended View Synthesis
> Todo: code coming soon!

### 3. NeRF Optimization with pose-conditioned dataset updates
> Todo: code coming soon!



# Inferencing
### 1. Render an image
> Todo: code coming soon!

### 2. Render a video
> Todo: code coming soon!



# Acknowledgement
This code is built upon [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch) implementation of [instant-ngp](https://nvlabs.github.io/instant-ngp/), and [Paddle](https://github.com/PaddlePaddle/Paddle) implementation of [DreamBooth](https://dreambooth.github.io/). 
<br>
We thank them for their nice implementation!
