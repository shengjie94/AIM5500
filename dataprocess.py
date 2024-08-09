import os
import re
from pathlib import Path
import random
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

class VideoFrameDataset:
    def __init__(self, video_dir, frames_per_video=5):
        self.video_dir = Path(video_dir)
        self.frames_per_video = frames_per_video
        self.video_files = list(self.video_dir.glob("*.mp4"))
        self.caption_files = list(self.video_dir.glob("*.srt"))
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for video_file in self.video_files:
            video_id = video_file.stem
            caption_file = self.video_dir / f"{video_id}.srt"
            if not caption_file.exists():
                continue
            captions = self._load_captions(caption_file)
            frames, timesteps = self._extract_frames(video_file)
            if frames and captions and timesteps:
                data.append({"frames": frames, "captions": captions, "timesteps": timesteps, "video_id": video_id})
        return data

    def _load_captions(self, caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            content = f.read()
        captions = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
        captions = captions.split('\n\n')
        captions = [caption.strip() for caption in captions if caption.strip()]
        return captions

    def _extract_frames(self, video_file):
        cap = cv2.VideoCapture(str(video_file))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [
            random.randint(0, total_frames // 8),
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4,
            random.randint(7 * total_frames // 8, total_frames - 1)
        ]

        timesteps = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timesteps.append(idx)

        cap.release()
        return frames if len(frames) == self.frames_per_video else None, timesteps if len(timesteps) == self.frames_per_video else None

    def split_data(self, train_ratio=0.7, val_ratio=0.1):
        train_data, test_data = train_test_split(self.data, train_size=train_ratio)
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_data, test_data = train_test_split(test_data, test_size=(1 - val_ratio_adjusted))
        return train_data, val_data, test_data

    def save_dataset(self, data, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i, item in enumerate(data):
            for j, frame in enumerate(item['frames']):
                frame = Image.fromarray(frame)
                frame.save(os.path.join(directory, f"{item['video_id']}_frame_{j}.png"))
            with open(os.path.join(directory, f"{item['video_id']}.txt"), 'w', encoding='utf-8') as f:
                f.write('\n'.join(item['captions']))
            with open(os.path.join(directory, f"{item['video_id']}_timesteps.txt"), 'w', encoding='utf-8') as f:
                f.write('\n'.join(map(str, item['timesteps'])))

video_dir = 'YU'
dataset = VideoFrameDataset(video_dir)

train_data, val_data, test_data = dataset.split_data()

dataset.save_dataset(train_data, 'YU/train')
dataset.save_dataset(val_data, 'YU/val')
dataset.save_dataset(test_data, 'YU/test')

print("Data processing completed successfully.")
