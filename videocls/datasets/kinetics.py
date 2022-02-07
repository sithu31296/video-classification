import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_video



class Kinetics(Dataset):
    def __init__(self, root: str, split='train', size: int = 256, transform: T.Compose = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.crop_size = size
        self.transform = transform

        root = Path(root) / split
        self.files = list(root.glob("*/*.mp4"))
        self.class_names = sorted([path.stem for path in root.iterdir()])
        self.num_classes = len(self.class_names)
        
        print(f"Num of videos in `{root}`: {len(self.files)}")
        print(f"Class Names: {self.class_names}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        label = self.class_names.index(file.parts[-2])
        frames, _, _ = read_video(str(file))
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, label


if __name__ == '__main__':
    dataset = Kinetics("data/k400", "val")
    frames, label = next(iter(dataset))
    print(frames.shape, label)