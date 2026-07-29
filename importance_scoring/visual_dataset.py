import json
import torch
from torch.utils.data import Dataset

class VisualImportanceDataset(Dataset):
    def __init__(self, visual_path, label_path):
        self.visual = torch.load(visual_path)
        with open(label_path) as f:
            self.labels = json.load(f)

        self.keys = []
        for k in self.visual.keys():
            # convert "video/seg.mp4" → "video_seg_000"
            base = k.split("/")[-1].replace(".mp4", "")
            label_key = base

            if label_key in self.labels:
                self.keys.append((k, label_key))

        print("✅ Dataset size:", len(self.keys))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        feat_key, label_key = self.keys[idx]
        v = self.visual[feat_key].float()
        y = torch.tensor(self.labels[label_key]).float()
        return v, y
