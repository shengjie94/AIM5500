import os
import torch
import time
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import random
import re
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchvision import transforms
import numpy as np

class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, tokenizer, transform=None, frames_per_video=5):
        self.video_dir = Path(video_dir)
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.tokenizer = tokenizer
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        video_ids = set([file.stem.rsplit('_frame', 1)[0] for file in self.video_dir.glob("*.png")])
        for video_id in video_ids:
            frames = self._load_frames(video_id)
            captions = self._load_captions(video_id)
            if frames and captions:
                data.append({"frames": frames, "captions": captions, "video_id": video_id})
        return data

    def _load_frames(self, video_id):
        frames = []
        for i in range(self.frames_per_video):
            frame_file = self.video_dir / f"{video_id}_frame_{i}.png"
            if frame_file.exists():
                frame = Image.open(frame_file)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                print(f"Missing frame: {frame_file}")
        return frames if len(frames) == self.frames_per_video else None

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
        captions = random.choice(item["captions"])

        if self.transform:
            frames = [self.transform(frame) if not isinstance(frame, torch.Tensor) else frame for frame in frames]

        tokenized_caption = self.tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return frames, tokenized_caption, item["video_id"]

def collate_fn(batch):
    frames, captions, video_ids = zip(*batch)
    frames = [frame for sublist in frames for frame in sublist]
    frames = torch.stack(frames, dim=0)
    input_ids = torch.cat([caption["input_ids"] for caption in captions])
    attention_mask = torch.cat([caption["attention_mask"] for caption in captions])
    return frames, {"input_ids": input_ids, "attention_mask": attention_mask}, video_ids

transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

test_dataset = VideoFrameDataset('/scratch/shengjie/svdtrain/YU/test', tokenizer, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder.to(device)
vae.to(device)
unet.to(device)

# Load the fine-tuned model weights
unet.load_state_dict(torch.load('fine-tuned-unet.pth'))
vae.load_state_dict(torch.load('fine-tuned-vae.pth'))

output_dir = Path('/scratch/shengjie/svdtrain/output')
output_dir.mkdir(parents=True, exist_ok=True)

noise_strength = 0.1  # Define noise_strength

def calculate_sigmas(timesteps, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod)**0.5
    return sqrt_one_minus_alphas_cumprod / sqrt_alphas_cumprod

def save_images(images, video_id, prefix):
    for i, img in enumerate(images):
        img = img.cpu().detach().numpy().transpose(1, 2, 0)
        img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)  # Unnormalize and convert to uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{video_id}_{prefix}_frame_{i}.png"), img)

print("Starting evaluation...")

unet.eval()
with torch.no_grad():
    for frames, captions, video_ids in tqdm(test_dataloader, desc="Evaluating", leave=False):
        frames = frames.to(device)
        input_ids = captions["input_ids"].to(device)
        attention_mask = captions["attention_mask"].to(device)
        video_id = video_ids[0]

        # Save original frames
        save_images(frames, video_id, "original")

        text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        if text_embeddings.size(-1) != 768 or text_embeddings.size(1) != frames.size(0):
            transform_layer = nn.Linear(text_embeddings.size(-1), 768).to(device)
            text_embeddings = transform_layer(text_embeddings)
        text_embeddings = text_embeddings.repeat(frames.size(0), 1, 1)

        # Only add noise for generating denoised images
        noise = torch.randn_like(frames) * noise_strength
        noisy_frames = frames + noise

        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (frames.shape[0],), device=device).long()
        sigmas = calculate_sigmas(timesteps, scheduler).to(device)

        latents = vae.encode(noisy_frames).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise_pred = unet(latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Ensure compatible shapes for denoising
        sigmas = sigmas[timesteps].view(-1, 1, 1, 1)
        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = noise_pred * c_out + c_skip * latents
        denoised_latents = F.interpolate(denoised_latents, size=(256, 256), mode='bilinear', align_corners=False)
        denoised_frames = denoised_latents[:, :3, :, :]  # Select the first 3 channels

        # Optional: Post-processing for denoised frames
        denoised_frames = denoised_frames.clamp(-1, 1)

        # Save denoised frames
        save_images(denoised_frames, video_id, "denoised")

print("Evaluation completed.")