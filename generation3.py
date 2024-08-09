import torch
import os
import cv2
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline, EulerDiscreteScheduler
from diffusers.utils import load_image, export_to_video
from PIL import Image

def load_and_resize_image(image_path, size=(1024, 576)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = Image.fromarray(image)
    return image

def extract_frames_from_video(video_path, output_dir, fps=12, duration=60):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    total_frames_needed = fps * duration
    frame_idx = 0
    frames_extracted = 0

    while frames_extracted < total_frames_needed:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frames_extracted}.png")
            cv2.imwrite(frame_path, frame)
            frames_extracted += 1
        
        frame_idx += 1

    cap.release()

# Directory paths
image_dir = '/scratch/shengjie/svdtrain/YU/test'
output_generated_dir = '/scratch/shengjie/svdtrain/YU/evalG'
original_video_dir = '/scratch/shengjie/svdtrain/YU'
output_original_dir = '/scratch/shengjie/svdtrain/YU/evalR'

# Load the pretrained models
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "/scratch/shengjie/svdtrain/fine-tuned-model-final/unet",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    "/scratch/shengjie/svdtrain/fine-tuned-model-final/vae",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
vision_model = CLIPVisionModelWithProjection.from_pretrained(
    "/scratch/shengjie/svdtrain/fine-tuned-model-final/vision_model",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
feature_extractor = CLIPImageProcessor.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    subfolder="feature_extractor",
    local_files_only=True,
)
scheduler = EulerDiscreteScheduler.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    subfolder="scheduler",
    local_files_only=True,
)

# Create the pipeline manually
pipe = StableVideoDiffusionPipeline(
    unet=unet,
    vae=vae,
    image_encoder=vision_model,
    feature_extractor=feature_extractor,
    scheduler=scheduler
)
pipe.to("cuda:0")

# Process each group of images in the directory
grouped_images = {}
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        group_name = '_'.join(filename.split('_')[:-1])
        if group_name not in grouped_images:
            grouped_images[group_name] = []
        grouped_images[group_name].append(filename)

for group_name, filenames in grouped_images.items():
    if len(filenames) != 5:
        print(f"Skipping group {group_name} due to insufficient frames.")
        continue
    
    images = [load_and_resize_image(os.path.join(image_dir, filename)) for filename in sorted(filenames)]
    
    generator = torch.manual_seed(-1)
    frames_per_image = 12  # Increase the number of frames per image to extend video duration
    all_frames = []

    # Generate frames for each image
    for image in images:
        with torch.inference_mode():
            frames = pipe(
                image,
                num_frames=frames_per_image,
                width=1024,
                height=576,
                decode_chunk_size=8,
                generator=generator,
                motion_bucket_id=127,
                fps=12,  # Adjust the fps if needed
                num_inference_steps=50  # Increase inference steps for better quality
            ).frames[0]
            all_frames.extend(frames)

    # Calculate the total number of frames and adjust fps for desired video length
    total_frames = len(all_frames)
    desired_length_seconds = 60
    adjusted_fps = total_frames / desired_length_seconds

    # Export the generated frames to a video
    video_path = os.path.join(output_generated_dir, f"{group_name}.mp4")
    export_to_video(all_frames, video_path, fps=adjusted_fps)
    
    # Extract frames from the original video
    original_video_path = os.path.join(original_video_dir, f"{group_name}.mp4")
    if os.path.exists(original_video_path):
        output_original_video_dir = os.path.join(output_original_dir, group_name)
        extract_frames_from_video(original_video_path, output_original_video_dir, fps=adjusted_fps, duration=desired_length_seconds)
    else:
        print(f"Original video not found for group {group_name}.")
