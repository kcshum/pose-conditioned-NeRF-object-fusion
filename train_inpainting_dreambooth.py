# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import contextlib
import gc
import hashlib
import itertools
import math
import os
import sys
import warnings
import random
from pathlib import Path
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import BatchSampler, DataLoader, Dataset, DistributedBatchSampler
from paddle.optimizer import AdamW
from paddle.vision import BaseTransform, transforms
from PIL import Image
from tqdm.auto import tqdm

from paddlenlp.trainer import set_seed
from paddlenlp.transformers import AutoTokenizer, PretrainedConfig
from paddlenlp.utils.log import logger
from ppdiffusers import (
    AutoencoderKL,
    DDPMScheduler,
    #DiffusionPipeline,
    UNet2DConditionModel,
    is_ppxformers_available,
)

from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from ppdiffusers.models.modeling_utils import freeze_params, unwrap_model
from ppdiffusers.optimization import get_scheduler


def url_or_path_join(*path_list):
    return os.path.join(*path_list) if os.path.isdir(os.path.join(*path_list)) else "/".join(path_list)


class Lambda(BaseTransform):
    def __init__(self, fn, keys=None):
        super().__init__(keys)
        self.fn = fn

    def _apply_image(self, img):
        return self.fn(img)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    try:
        text_encoder_config = PretrainedConfig.from_pretrained(
            url_or_path_join(pretrained_model_name_or_path, "text_encoder")
        )
        model_class = text_encoder_config.architectures[0]
    except Exception:
        model_class = "LDMBertModel"
    if model_class == "CLIPTextModel":
        from paddlenlp.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "BertModel":
        from paddlenlp.transformers import BertModel

        return BertModel
    elif model_class == "LDMBertModel":
        from ppdiffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import (
            LDMBertModel,
        )

        return LDMBertModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def set_recompute(model, value=False):
    def fn(layer):
        # ldmbert
        if hasattr(layer, "enable_recompute"):
            layer.enable_recompute = value
            print("Set", layer.__class__, "recompute", layer.enable_recompute)
        # unet
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = value
            print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

    model.apply(fn)


def get_report_to(args):
    if args.report_to == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logging_dir)
    elif args.report_to == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("report_to must be in ['visualdl', 'tensorboard']")
    return writer


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training dreambooth script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pre-trained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='dataset',
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--object_data",
        type=str,
        default=None,
        required=True,
        help="The name of instance data folder.",
    )
    parser.add_argument(
        "--background_data",
        type=str,
        default=None,
        required=True,
        help="The name of background data folder.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--object_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the object.",
    )
    parser.add_argument(
        "--background_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the background.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dream_outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1, help="Batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--max_train_steps_OBJ",
        type=int,
        default=4000,
        help="Total number of training steps for OBJ to perform.",
    )
    parser.add_argument(
        "--max_train_steps_BG",
        type=int,
        default=400,
        help="Total number of training steps for BG to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) or [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl) log directory. Will default to"
            "*output_dir/logs"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="visualdl",
        choices=["tensorboard", "visualdl"],
        help="Log writer type.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50000,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--small",
        type=float,
        default=0.2,
        help="Random mask top-left coordinate, proportional to resolution.",
    )
    parser.add_argument(
        "--big",
        type=float,
        default=0.8,
        help="Random mask bottom-right coordinate, proportional to resolution.",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.1,
        help="Random mask jitter of top-left and bottom-right coordinate, proportional to resolution.",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.object_data is None:
        raise ValueError("You must specify a train object data.")
    if args.background_data is None:
        raise ValueError("You must specify a train background data.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    args.output_dir = os.path.join(args.output_dir, args.object_data + '_and_' + args.background_data)

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if args.height is None or args.width is None and args.resolution is not None:
        args.height = args.width = args.resolution

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        height=512,
        width=512,
        center_crop=False,
        interpolation="bilinear",
        random_flip=False,
    ):
        self.height = height
        self.width = width
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        ext = ["png", "jpg", "jpeg", "bmp", "PNG", "JPG", "JPEG", "BMP"]
        self.instance_images_path = []
        for p in Path(instance_data_root).iterdir():
            if any(suffix in p.name for suffix in ext):
                self.instance_images_path.append(p)
        self.instance_images_path.sort()

        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = []
            for p in Path(class_data_root).iterdir():
                if any(suffix in p.name for suffix in ext):
                    self.class_images_path.append(p)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=interpolation),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        this_prompt = 'a ' + self.instance_prompt
        example["instance_prompt_ids"] = self.tokenizer(
            this_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=False,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_attention_mask=False,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    return f"{organization}/{model_id}"


def main():
    args = parse_args()
    rank = paddle.distributed.get_rank()
    is_main_process = rank == 0
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                safety_checker=None,
            )
            if args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
                try:
                    pipeline.unet.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warn(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            batch_sampler = (
                DistributedBatchSampler(sample_dataset, batch_size=args.sample_batch_size, shuffle=False)
                if num_processes > 1
                else BatchSampler(sample_dataset, batch_size=args.sample_batch_size, shuffle=False)
            )
            sample_dataloader = DataLoader(
                sample_dataset, batch_sampler=batch_sampler, num_workers=args.dataloader_num_workers
            )

            for example in tqdm(sample_dataloader, desc="Generating class images", disable=not is_main_process):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
            pipeline.to("cpu")
            del pipeline
            gc.collect()

    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(url_or_path_join(args.pretrained_model_name_or_path, "tokenizer"))

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder")
    )
    text_config = text_encoder.config if isinstance(text_encoder.config, dict) else text_encoder.config.to_dict()
    if text_config.get("use_attention_mask", None) is not None and text_config["use_attention_mask"]:
        use_attention_mask = True
    else:
        use_attention_mask = False
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    freeze_params(vae.parameters())
    if not args.train_text_encoder:
        freeze_params(text_encoder.parameters())
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            set_recompute(text_encoder, True)

    if args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warn(
                "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                f" correctly and a GPU is available: {e}"
            )

    train_OBJ_dataset = DreamBoothDataset(
        instance_data_root=os.path.join(args.data_dir, 'object', args.object_data),
        instance_prompt=args.object_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        interpolation="bilinear",
        random_flip=args.random_flip,
    )

    train_BG_dataset = DreamBoothDataset(
        instance_data_root=os.path.join(args.data_dir, 'background', args.background_data, 'images'),
        instance_prompt=args.background_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        interpolation="bilinear",
        random_flip=args.random_flip,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = paddle.stack(pixel_values).astype("float32")

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pd"
        ).input_ids

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

    train_sampler = (
        DistributedBatchSampler(train_OBJ_dataset, batch_size=args.train_batch_size, shuffle=True)
        if num_processes > 1
        else BatchSampler(train_OBJ_dataset, batch_size=args.train_batch_size, shuffle=True)
    )
    train_dataloader = DataLoader(
        train_OBJ_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=args.dataloader_num_workers
    )

    train_BG_sampler = (
        DistributedBatchSampler(train_BG_dataset, batch_size=args.train_batch_size, shuffle=True)
        if num_processes > 1
        else BatchSampler(train_BG_dataset, batch_size=args.train_batch_size, shuffle=True)
    )
    train_BG_dataloader = DataLoader(
        train_BG_dataset, batch_sampler=train_BG_sampler, collate_fn=collate_fn, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    args.gradient_accumulation_steps = 1

    num_update_steps_per_epoch_OBJ = math.ceil((len(train_dataloader)) / args.gradient_accumulation_steps)
    num_train_epochs_OBJ = math.ceil(args.max_train_steps_OBJ / num_update_steps_per_epoch_OBJ)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch_BG = math.ceil((len(train_BG_dataloader)) / args.gradient_accumulation_steps)
    num_train_epochs_BG = math.ceil(args.max_train_steps_BG / num_update_steps_per_epoch_BG)

    if num_processes > 1:
        unet = paddle.DataParallel(unet)
        if args.train_text_encoder:
            text_encoder = paddle.DataParallel(text_encoder)

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps_OBJ * args.gradient_accumulation_steps + args.max_train_steps_BG * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Initialize the optimizer
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=params_to_optimize,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
    )

    if is_main_process:
        logger.info("-----------  Configuration Arguments -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
        writer = get_report_to(args)

    # Train!
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_OBJ_dataset) + len(train_BG_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num OBJ Epochs = {num_train_epochs_OBJ}")
    logger.info(f"  Num BG Epochs = {num_train_epochs_BG}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps OBJ = {args.max_train_steps_OBJ}")
    logger.info(f"  Total optimization steps BG = {args.max_train_steps_BG}")

    # Only show the progress bar once on each machine.
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps_OBJ + args.max_train_steps_BG), disable=not is_main_process)
    progress_bar.set_description("Train Steps")

    # Keep vae in eval model as we don't train these
    vae.eval()
    if args.train_text_encoder:
       text_encoder.train()
    else:
       text_encoder.eval()
    unet.train()

    def prepare_mask_and_masked_image(image):
        mask = paddle.zeros(shape=[batch_size, 1, args.resolution, args.resolution])

        small = int(args.resolution * args.small)
        big = int(args.resolution * args.big)
        random_gap = int(args.resolution * args.gap)

        mask[:, :,
        small + random.randint(-random_gap, random_gap):big + random.randint(-random_gap, random_gap),
        small + random.randint(-random_gap, random_gap):big + random.randint(-random_gap, random_gap)] = 1

        if isinstance(image, paddle.Tensor):
            assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
            assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
            assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")

            # Check mask is in [0, 1]
            if mask.min() < 0 or mask.max() > 1:
                raise ValueError("Mask should be in [0, 1] range")

            # Binarize mask
            mask = paddle.where(mask < 0.5, 0.0, 1.0)

            # Image as float32
            image = image.cast(paddle.float32)

        masked_image = image * (mask < 0.5)

        return mask, masked_image

    def prepare_mask_latents(mask, masked_image, batch_size, height, width, dtype):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        mask = F.interpolate(mask, size=(height // vae_scale_factor, width // vae_scale_factor))
        mask = mask.cast(dtype)

        masked_image = masked_image.cast(dtype)

        # encode the mask image into latents space so we can concatenate it to the latents

        masked_image_latents = vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = vae.config.scaling_factor * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.tile([batch_size // mask.shape[0], 1, 1, 1])
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.tile([batch_size // masked_image_latents.shape[0], 1, 1, 1])

        mask = mask
        masked_image_latents = (
            masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.cast(dtype)
        return mask, masked_image_latents

    for max_train_steps_OBJ, train_X_dataloader, num_train_epochs_X in zip([args.max_train_steps_OBJ, args.max_train_steps_BG],
                                                                       [train_dataloader, train_BG_dataloader],
                                                                       [num_train_epochs_OBJ, num_train_epochs_BG]):
        for epoch in range(num_train_epochs_X):
            for step, batch in enumerate(train_X_dataloader):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = paddle.randn(latents.shape)
                batch_size = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = paddle.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).cast("int64")

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                mask, masked_image = prepare_mask_and_masked_image(batch["pixel_values"])

                if num_processes > 1 and (
                    args.gradient_checkpointing or ((step + 1) % args.gradient_accumulation_steps != 0)
                ):
                    # grad acc, no_sync when (step + 1) % args.gradient_accumulation_steps != 0:
                    # gradient_checkpointing, no_sync every where
                    # gradient_checkpointing + grad_acc, no_sync every where
                    unet_ctx_manager = unet.no_sync()
                    if args.train_text_encoder:
                        text_encoder_ctx_manager = text_encoder.no_sync()
                    else:
                        text_encoder_ctx_manager = (
                            contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                        )
                else:
                    unet_ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                    text_encoder_ctx_manager = (
                        contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                    )

                with text_encoder_ctx_manager:
                    # Get the text embedding for conditioning
                    if use_attention_mask:
                        attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).cast("int64")
                    else:
                        attention_mask = None
                    encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=attention_mask)[0]

                    mask, masked_image_latents = prepare_mask_latents(
                        mask,
                        masked_image,
                        batch_size,
                        args.resolution,
                        args.resolution,
                        encoder_hidden_states.dtype,
                    )

                    num_channels_latents = vae.config.latent_channels
                    num_channels_mask = mask.shape[1]
                    num_channels_masked_image = masked_image_latents.shape[1]
                    if num_channels_latents + num_channels_mask + num_channels_masked_image != unet.config.in_channels:
                        raise ValueError(
                            f"Incorrect configuration settings! The config of `pipeline.unet`: {unet.config} expects"
                            f" {unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                            f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                            f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                            " `pipeline.unet` or your `mask_image` or `image` input."
                        )

                    with unet_ctx_manager:
                        latent_model_input = paddle.concat([noisy_latents, mask, masked_image_latents], axis=1)
                        # Predict the noise residual / sample
                        model_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        if args.with_prior_preservation:
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred, model_pred_prior = model_pred.chunk(2, axis=0)
                            target, target_prior = target.chunk(2, axis=0)

                            # Compute instance loss
                            loss = F.mse_loss(model_pred, target, reduction="mean")

                            # Compute prior loss
                            prior_loss = F.mse_loss(model_pred_prior, target_prior, reduction="mean")

                            # Add the prior loss to the instance loss.
                            loss = loss + args.prior_loss_weight * prior_loss
                        else:
                            loss = F.mse_loss(model_pred, target, reduction="mean")

                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if num_processes > 1 and args.gradient_checkpointing:
                        fused_allreduce_gradients(params_to_optimize, None)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.clear_grad()
                    progress_bar.update(1)
                    global_step += 1
                    step_loss = loss.item() * args.gradient_accumulation_steps
                    logs = {
                        "epoch": str(epoch).zfill(4),
                        "step_loss": round(step_loss, 10),
                        "lr": lr_scheduler.get_lr(),
                    }
                    progress_bar.set_postfix(**logs)

                    if is_main_process:
                        for name, val in logs.items():
                            if name == "epoch":
                                continue
                            writer.add_scalar(f"train/{name}", val, global_step)

                        if global_step % args.checkpointing_steps == 0:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
                            if args.train_text_encoder:
                                unwrap_model(text_encoder).save_pretrained(os.path.join(save_path, "text_encoder"))

                    if global_step >= (int(args.max_train_steps_OBJ) + int(args.max_train_steps_BG)):
                        break

    # Create the pipeline using the trained modules and save it.
    if is_main_process:
        writer.close()
        # Create the pipeline using using the trained modules and save it.
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrap_model(unet),
            text_encoder=unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
