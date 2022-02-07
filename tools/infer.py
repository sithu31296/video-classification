import torch
import argparse
from torchvision.io import read_video
from torchvision import transforms as T
from pytorchvideo.transforms import (
    Permute,
    ShortSideScale,
    Div255,
    UniformTemporalSubsample,
    Normalize
)
from tabulate import tabulate
from videocls.models import *


def load_labels(model):
    if isinstance(model, UniFormer):
        label_path = 'data/uniformer_k400_categories.txt'
    else:
        label_path = 'data/k400_classnames.txt'
    
    with open(label_path) as f:
        class_names = f.read().splitlines()
    return class_names


class VideoClassification:
    def __init__(self, model: str, model_path: str, num_classes: int) -> None:
        model_name, variant = model.split('-')
        self.size = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = eval(model_name)(variant, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.class_names = load_labels(self.model)

        self.transform = T.Compose([
            Permute([3, 0, 1, 2]),
            UniformTemporalSubsample(8),
            Div255(),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ShortSideScale(self.size),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def __call__(self, video_path):
        frames, _, _ = read_video(video_path)
        frames = self.transform(frames)
        frames = frames.to(self.device)
        with torch.inference_mode():
            preds = self.model(frames)
        preds = preds.softmax(dim=-1).flatten()
        probs, indices = torch.topk(preds, 5)
        return [(self.class_names[int(index)], f"{p.item() * 100:.2f}") for index, p in zip(indices, probs)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/archers.mp4', help="video file name to be tested")
    parser.add_argument('--model', type=str, default='UniFormer-S', help="model name (cases should match with model constructor)")
    parser.add_argument('--model_path', type=str, default='C:\\Users\\sithu\\Documents\\weights\\videocls\\uniformer\\uniformer_small_k400_8x8.pth', help='model weights file')
    parser.add_argument('--num_classes', type=int, default=400, help="number of classes used to train the model")
    args = vars(parser.parse_args())
    source = args.pop('source')
    classifier = VideoClassification(**args)
    classes = classifier(source)
    print(tabulate(classes, headers=['Class', 'Score (%)']))
