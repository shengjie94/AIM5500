import os
import time
from pathlib import Path
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
import PIL
from PIL import Image
import numpy as np
import cv2
import shutil
import logging
import math
from tqdm.auto import tqdm

import sys
sys.path.append('/scratch/shengjie/svdtrain/src')
from unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

logger = get_logger(__name__, log_level="INFO")

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

device = set_device()

class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, processor, frames_per_video=5):
        self.video_dir = Path(video_dir)
        self.frames_per_video = frames_per_video
        self.processor = processor
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        video_ids = set([file.stem.rsplit('_frame', 1)[0] for file in self.video_dir.glob("*.png")])
        for video_id in video_ids:
            frames = self._load_frames(video_id)
            captions = self._load_captions(video_id)
            if frames and captions:
                for frame, caption in zip(frames, captions):
                    data.append({"frames": frames, "caption": caption, "video_id": video_id})
        return data

    def _load_frames(self, video_id):
        frames = []
        for idx in range(self.frames_per_video):
            frame_file = self.video_dir / f"{video_id}_frame_{idx}.png"
            frame = Image.open(frame_file)
            frame = frame.convert("RGB")
            frame = np.array(frame).astype(np.float32) / 255.0
            frames.append(frame)
        return frames

    def _load_captions(self, video_id):
        caption_file = self.video_dir / f"{video_id}.txt"
        if caption_file.exists():
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = f.read().split('\n')
            return [caption.strip() for caption in captions if caption.strip()]
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = item["frames"]
        caption = item["caption"]
        pixel_values_list = []
        for frame in frames:
            inputs = self.processor(images=frame, return_tensors="pt", do_rescale=False)
            pixel_values_list.append(inputs["pixel_values"])
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return {"pixel_values": pixel_values, "caption": caption}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0).to(device)
    noise = torch.randn_like(pixel_values) * noise_strength
    noisy_pixel_values = pixel_values + noise

    combined_pixel_values = torch.cat([pixel_values, noisy_pixel_values], dim=2)  # Combine along the channel dimension

    captions = [item["caption"] for item in batch]
    return {"pixel_values": combined_pixel_values, "captions": captions}

processor = CLIPImageProcessor.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="feature_extractor")
train_dataset = VideoFrameDataset('/scratch/shengjie/svdtrain/YU/train', processor)
val_dataset = VideoFrameDataset('/scratch/shengjie/svdtrain/YU/val', processor)
test_dataset = VideoFrameDataset('/scratch/shengjie/svdtrain/YU/test', processor)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

vision_model = CLIPVisionModelWithProjection.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="image_encoder")
vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="vae")
unet = UNetSpatioTemporalConditionModel.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="unet")

scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="scheduler")

optimizer = AdamW(unet.parameters(), lr=1e-5)
scaler = GradScaler()

# EMA model
ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

accelerator = Accelerator(mixed_precision='no')

train_dataloader, val_dataloader, test_dataloader, unet, optimizer, scaler, vision_model, vae = accelerator.prepare(
    train_dataloader, val_dataloader, test_dataloader, unet, optimizer, scaler, vision_model, vae
)

num_epochs = 100
accumulation_steps = 32
noise_strength = 0.05

def tensor_to_vae_latent(t, vae):
    latents = vae.encode(t).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents

def calculate_sigmas(timesteps, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod)**0.5
    return sqrt_one_minus_alphas_cumprod[timesteps] / sqrt_alphas_cumprod[timesteps]

def get_added_time_ids(fps, noise_aug_strength, batch_size, device, dtype):
    add_time_ids = torch.tensor([[fps, 127, noise_aug_strength]], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

def encode_image(pixel_values):
    batch_size, frames_per_video, channels, height, width = pixel_values.shape
    pixel_values = pixel_values[:, :, :3, :, :].contiguous()
    pixel_values = pixel_values.view(batch_size * frames_per_video, 3, height, width)
    pixel_values = pixel_values.to(device)
    pixel_values = F.interpolate(pixel_values, size=(224, 224), mode='bilinear', align_corners=False)
    vision_outputs = vision_model(pixel_values=pixel_values)
    image_embeddings = vision_outputs.image_embeds
    image_embeddings = image_embeddings.view(batch_size, frames_per_video, -1)
    return image_embeddings

def create_8_channel_input(pixel_values, conditional_pixel_values, vae):
    pixel_values = pixel_values[:, :, :3, :, :].contiguous()
    conditional_pixel_values = conditional_pixel_values[:, :, :3, :, :].contiguous()
    batch_size, frames_per_video, channels, height, width = pixel_values.shape
    pixel_values = pixel_values.view(batch_size * frames_per_video, channels, height, width)
    conditional_pixel_values = conditional_pixel_values.view(batch_size * frames_per_video, channels, height, width)
    latents = tensor_to_vae_latent(pixel_values, vae).to(device)
    conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae).to(device)
    latents = F.interpolate(latents, size=(48, 48), mode='bilinear', align_corners=False)
    conditional_latents = F.interpolate(conditional_latents, size=(48, 48), mode='bilinear', align_corners=False)
    latents = latents.view(batch_size, frames_per_video, -1, 48, 48)
    conditional_latents = conditional_latents.view(batch_size, frames_per_video, -1, 48, 48)
    combined_latents = torch.cat((latents, conditional_latents), dim=2)
    return combined_latents

def evaluate_model(dataloader, model, vision_model, vae, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs in dataloader:
            pixel_values = inputs["pixel_values"].to(device)
            captions = inputs["captions"]
            image_embeddings = encode_image(pixel_values).to(device)
            noise = torch.randn_like(pixel_values) * noise_strength
            conditional_pixel_values = pixel_values + noise
            combined_latents = create_8_channel_input(pixel_values, conditional_pixel_values, vae).to(device)
            added_time_ids = get_added_time_ids(fps=7, noise_aug_strength=0.1, batch_size=pixel_values.size(0), device=device, dtype=combined_latents.dtype)
            sigmas = torch.rand([combined_latents.shape[0], 1, 1, 1], dtype=combined_latents.dtype, device=combined_latents.device)
            noisy_latents = combined_latents + torch.randn_like(combined_latents) * sigmas
            timesteps = torch.Tensor([[0.25 * sigma.log() for sigma in batch] for batch in sigmas]).to(device).long()
            timesteps = timesteps.view(-1)

            noise_pred = model(noisy_latents, timesteps, encoder_hidden_states=image_embeddings, added_time_ids=added_time_ids).sample
            sigmas = calculate_sigmas(timesteps, scheduler).view(combined_latents.size(0), 1, 1, 1).to(device)
            c_out = -sigmas / ((sigmas**2 + 1)**0.5)
            c_skip = 1 / (sigmas**2 + 1)
            combined_latents_4_channel = combined_latents[:, :, :4, :, :]
            denoised_latents = noise_pred * c_out + c_skip * combined_latents_4_channel
            batch_size, frames_per_video, _, height, width = pixel_values.shape
            denoised_latents = denoised_latents.view(batch_size * frames_per_video, -1, 48, 48)
            denoised_frames = denoised_latents[:, :3, :, :]
            denoised_frames_resized = F.interpolate(denoised_frames, size=(height, width), mode='bilinear', align_corners=False)
            denoised_frames_resized = denoised_frames_resized.view(batch_size, frames_per_video, 3, height, width)

            loss = F.mse_loss(denoised_frames_resized.float(), pixel_values[:, :, :3, :, :].float())

            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss

for epoch in range(num_epochs):
    total_loss = 0.0
    epoch_start_time = time.time()
    optimizer.zero_grad()

    for i, inputs in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
        pixel_values = inputs["pixel_values"].to(device)
        captions = inputs["captions"]
        
        with torch.no_grad():
            image_embeddings = encode_image(pixel_values).to(device)
        
        noise = torch.randn_like(pixel_values) * noise_strength
        conditional_pixel_values = pixel_values + noise
        combined_latents = create_8_channel_input(pixel_values, conditional_pixel_values, vae).to(device)
        added_time_ids = get_added_time_ids(fps=7, noise_aug_strength=0.1, batch_size=pixel_values.size(0), device=device, dtype=combined_latents.dtype)
        sigmas = torch.rand([combined_latents.shape[0], 1, 1, 1], dtype=combined_latents.dtype, device=combined_latents.device)
        noisy_latents = combined_latents + torch.randn_like(combined_latents) * sigmas
        timesteps = torch.Tensor([[0.25 * sigma.log() for sigma in batch] for batch in sigmas]).to(device).long()
        timesteps = timesteps.view(-1)

        try:
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=image_embeddings, added_time_ids=added_time_ids).sample
        except Exception as e:
            print("Error during UNet forward pass:", str(e))
            raise e

        sigmas = calculate_sigmas(timesteps, scheduler).view(combined_latents.size(0), 1, 1, 1).to(device)
        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)

        combined_latents_4_channel = combined_latents[:, :, :4, :, :]
        denoised_latents = noise_pred * c_out + c_skip * combined_latents_4_channel

        batch_size, frames_per_video, _, height, width = pixel_values.shape
        denoised_latents = denoised_latents.view(batch_size * frames_per_video, -1, 48, 48)

        denoised_frames = denoised_latents[:, :3, :, :]
        denoised_frames_resized = F.interpolate(denoised_frames, size=(height, width), mode='bilinear', align_corners=False)
        denoised_frames_resized = denoised_frames_resized.view(batch_size, frames_per_video, 3, height, width)

        loss = F.mse_loss(denoised_frames_resized.float(), pixel_values[:, :, :3, :, :].float())

        # Ensure loss is properly scaled
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            # Unscale the gradients before clipping and optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}: {100 * (epoch + 1) / num_epochs:.2f}% | Avg. Train Loss: {avg_train_loss:.4f} | Time: {time.time() - epoch_start_time:.2f}s")
    avg_val_loss = evaluate_model(val_dataloader, unet, vision_model, vae, device)
    print(f"Epoch {epoch + 1}/{num_epochs}: {100 * (epoch + 1) / num_epochs:.2f}% | Avg. Val Loss: {avg_val_loss:.4f} | Time: {time.time() - epoch_start_time:.2f}s")

    if (epoch + 1) % 10 == 0:
        accelerator.wait_for_everyone()
        unet = accelerator.unwrap_model(unet)
        vae = accelerator.unwrap_model(vae)
        vision_model = accelerator.unwrap_model(vision_model)
        
        save_dir = f"fine-tuned-model-epoch-{epoch + 1}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        unet.save_pretrained(os.path.join(save_dir, "unet"))
        vae.save_pretrained(os.path.join(save_dir, "vae"))
        vision_model.save_pretrained(os.path.join(save_dir, "vision_model"))

accelerator.wait_for_everyone()
unet = accelerator.unwrap_model(unet)
vae = accelerator.unwrap_model(vae)
vision_model = accelerator.unwrap_model(vision_model)

final_save_dir = 'fine-tuned-model-final'
if not os.path.exists(final_save_dir):
    os.makedirs(final_save_dir)

unet.save_pretrained(os.path.join(final_save_dir, "unet"))
vae.save_pretrained(os.path.join(final_save_dir, "vae"))
vision_model.save_pretrained(os.path.join(final_save_dir, "vision_model"))

print("Training completed.")
