import os
import cv2
import numpy as np
import keras
import albumentations as A
import segmentation_models as sm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Define preprocessing function for the model
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(
            image=lambda x, **kwargs: preprocessing_fn(x),
            mask=lambda x, **kwargs: x,
        ),
        A.Resize(352, 352)
    ]
    return A.Compose(_transform)

def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def visualize(image, gt_mask, pr_mask, output_path):
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')
    plt.imshow(denormalize(image))
    
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth Mask')
    if gt_mask is not None:
        plt.imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    else:
        plt.text(0.5, 0.5, 'No GT Mask', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted Mask')
    plt.imshow(pr_mask, cmap='gray', vmin=0, vmax=1)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Load the trained model
def load_model(model_path: str) -> keras.Model:
    model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    model.load_weights(model_path)
    print("Model loaded successfully")
    return model

# Inference class to handle model predictions
class InferenceModel:
    def __init__(self, model: keras.Model, input_size: Tuple[int, int]):
        self.model = model
        self.input_size = input_size
        self.preprocessing = get_preprocessing(preprocess_input)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sample = self.preprocessing(image=image)
        image = sample['image']
        image = np.expand_dims(image, axis=0)
        mask = self.model.predict(image, verbose=0)
        mask = mask.round().squeeze()
        return mask

# Dataset class for loading images and masks
class Dataset:
    CLASSES = ['smoke']
    
    def __init__(self, images_dir, masks_dir=None, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [str(Path(images_dir) / image_id) for image_id in self.ids]
        self.masks_fps = []
        self.has_masks = masks_dir is not None and os.path.exists(masks_dir)
        
        if self.has_masks:
            mask_files = os.listdir(masks_dir)
            for image_id in self.ids:
                base_name = os.path.splitext(image_id)[0]
                matching_mask = next((mask for mask in mask_files if base_name in mask), None)
                if matching_mask:
                    self.masks_fps.append(str(Path(masks_dir) / matching_mask))
                else:
                    self.masks_fps.append(None)
        else:
            self.masks_fps = [None] * len(self.ids)
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] if classes else [0, 1]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = None
        if self.masks_fps[i] is not None:
            mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
            binary_mask = np.zeros_like(mask, dtype=np.float32)
            binary_mask[mask >= 248] = 1.0
            mask = binary_mask[..., np.newaxis]
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask if mask is not None else np.zeros_like(image))
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask if mask is not None else np.zeros_like(image))
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        return len(self.ids)

# MediaProcessor class
class MediaProcessor:
    def __init__(self, model: InferenceModel, images_dir: str, videos_dir: str, save_output: bool, input_size: Tuple[int, int], num_classes: int, masks_dir: str = None):
        self.model = model
        self.images_dir = images_dir
        self.videos_dir = videos_dir
        self.save_output = save_output
        self.input_size = input_size
        self.num_classes = num_classes
        self.masks_dir = masks_dir
        self.cap = None
        self.writer = None
        self.output_path = None

    def _is_image_input(self, source: str) -> bool:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return any(source.lower().endswith(ext) for ext in image_extensions)

    def _is_video_input(self, source: str) -> bool:
        video_extensions = ['.mp4', '.avi', '.mov']
        return any(source.lower().endswith(ext) for ext in video_extensions)

    def _init_capture(self, video_path: str) -> cv2.VideoCapture:
        return cv2.VideoCapture(video_path)

    def _init_writer(self, input_source: str, output_dir: Path, is_image: bool, size: Tuple[int, int] = None):
        input_name = Path(input_source).stem
        if is_image:
            self.output_path = str(output_dir / f'{input_name}_output.jpg')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
            self.output_path = str(output_dir / f'{input_name}_output.mp4')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, size or self.input_size)

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == 1] = (0, 0, 255)
        return colored_mask

    def _process_image(self, image_path: str, output_dir: Path):
        dataset = Dataset(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            classes=['smoke'],
            augmentation=A.Compose([A.Resize(352, 352)]),
            preprocessing=get_preprocessing(preprocess_input)
        )
        normalized_image_path = str(Path(image_path))
        if normalized_image_path not in dataset.images_fps:
            print(f"Error: Image {normalized_image_path} not in dataset.images_fps")
            return
        image_idx = dataset.images_fps.index(normalized_image_path)
        
        image, gt_mask = dataset[image_idx]
        
        frame = cv2.imread(image_path)
        pr_mask = self.model.infer(frame)
        
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask.squeeze(), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        pr_mask = cv2.resize(pr_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if self.save_output:
            self._init_writer(image_path, output_dir, is_image=True)
            visualize(image, gt_mask, pr_mask, self.output_path)

    def _process_video(self, video_path: str, output_dir: Path):
        self.cap = self._init_capture(video_path)
        if self.save_output:
            self._init_writer(video_path, output_dir, is_image=False)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            mask = self.model.infer(frame)
            mask_colored = self._colorize_mask(mask)
            mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            if self.writer:
                self.writer.write(cv2.resize(overlay, self.input_size))

        self._cleanup()

    def process(self, output_dir: Path):
        if os.path.exists(self.images_dir):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for file_path in Path(self.images_dir).iterdir():
                file_str = str(file_path)
                if any(file_str.lower().endswith(ext) for ext in image_extensions):
                    self._process_image(file_str, output_dir)
        else:
            print(f"Images directory not found: {self.images_dir}")

        if os.path.exists(self.videos_dir):
            video_extensions = ['.mp4', '.avi', '.mov']
            for file_path in Path(self.videos_dir).iterdir():
                file_str = str(file_path)
                if any(file_str.lower().endswith(ext) for ext in video_extensions):
                    self._process_video(file_str, output_dir)
        else:
            print(f"Videos directory not found: {self.videos_dir}")

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

# Main execution
if __name__ == "__main__":
    base_dir = "./"
    model_path = os.path.join(base_dir, "output/exp12/best_model.h5")
    input_base = os.path.join(base_dir, "test_inference")
    images_dir = os.path.join(input_base, "images/images")
    masks_dir = os.path.join(input_base, "images/masks")
    videos_dir = os.path.join(input_base, "videos")
    output_dir = os.path.join(base_dir, "output/inference")
    os.makedirs(output_dir, exist_ok=True)
    input_size = (352, 352)
    num_classes = 1

    model = load_model(model_path)
    inference_model = InferenceModel(model, input_size)
    processor = MediaProcessor(
        model=inference_model,
        images_dir=images_dir,
        videos_dir=videos_dir,
        save_output=True,
        input_size=input_size,
        num_classes=num_classes,
        masks_dir=masks_dir
    )
    processor.process(Path(output_dir))