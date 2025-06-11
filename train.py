import os
import cv2
import keras
import numpy as np
import albumentations as A
import segmentation_models as sm
from pathlib import Path
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, images_dir, masks_dir, label_to_pixel, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = []
        self.label_to_pixel = label_to_pixel
        self.num_classes = 2  # Binary: background + smoke
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        mask_files = os.listdir(masks_dir)
        
        for image_id in self.ids:
            base_name = os.path.splitext(image_id)[0]
            
            mask_path = None
            for mask_file in mask_files:
                if mask_file.startswith(base_name) and mask_file.endswith(('.png', '.jpg', '.bmp')):
                    mask_path = os.path.join(masks_dir, mask_file)
                    break
            
            if mask_path is None:
                print(f"Warning: No mask found for image {image_id}")
                continue
            
            self.masks_fps.append(mask_path)
        
        self.ids = [os.path.basename(img_path) for img_path in self.images_fps[:len(self.masks_fps)]]
        self.images_fps = self.images_fps[:len(self.masks_fps)]

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        
        # Verify image and mask dimensions
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image and mask dimensions mismatch for {self.images_fps[i]}: "
                             f"image={image.shape[:2]}, mask={mask.shape[:2]}")
        
        # Convert mask to binary
        mask = (mask == self.label_to_pixel['smoke']).astype(np.float32)[..., np.newaxis]

        # Apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # Verify post-augmentation dimensions
            if image.shape[:2] != (320, 320):
                raise ValueError(f"Post-augmentation image size is {image.shape[:2]}, expected (320, 320)")

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask

    def __len__(self):
        return len(self.ids)

class DataLoader(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = min((i + 1) * self.batch_size, len(self.dataset))
        data = [self.dataset[j] for j in range(start, stop)]
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

class ModelTrainer:
    def __init__(self, backbone, num_classes, lr, epochs, output_dir, pretrained_weights=None):
        self.n_classes = num_classes
        self.model = sm.Unet(
            backbone,
            classes=1,  # Binary segmentation
            activation='sigmoid'
        )
        if pretrained_weights:
            Path(pretrained_weights).parent.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(pretrained_weights):
                raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights}. ")
            print(f"Loading pretrained weights from {pretrained_weights}")
            self.model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
        self.optim = keras.optimizers.Adam(lr)
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        self.model.compile(self.optim, total_loss, metrics)
        self.epochs = epochs
        self.output_dir = output_dir
        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(Path(output_dir) / 'best_model.h5'), save_weights_only=True, save_best_only=True, mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                str(Path(output_dir) / 'last_model.h5'), save_weights_only=True, save_best_only=False
            ),
            keras.callbacks.ReduceLROnPlateau()
        ]

    def train(self, train_dataloader, valid_dataloader):
        history = self.model.fit(
            train_dataloader,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=valid_dataloader,
            steps_per_epoch=len(train_dataloader),
            validation_steps=len(valid_dataloader)
        )
        return history

    def evaluate(self, test_dataloader):
        scores = self.model.evaluate(test_dataloader)
        return scores

    def predict(self, image):
        image = np.expand_dims(image, axis=0)
        pr_mask = self.model.predict(image)
        return pr_mask.round()

class ResultSaver:
    def __init__(self, label_to_pixel):
        self.label_to_pixel = label_to_pixel
        self.num_classes = 2  # Binary

    @staticmethod
    def denormalize(x):
        x_max = np.percentile(x, 98)
        x_min = np.percentile(x, 2)
        x = (x - x_min) / (x_max - x_min)
        return x.clip(0, 1)

    def save_results(self, test_dataset, model_trainer, output_dir):
        for i in range(len(test_dataset)):
            image, gt_mask = test_dataset[i]
            pr_mask = model_trainer.predict(image)
            image = self.denormalize(image)
            
            gt_mask = gt_mask[..., 0].squeeze()
            pr_mask = pr_mask[..., 0].squeeze()

            # Create figure with three subplots
            plt.figure(figsize=(15, 5))
            
            # Input image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Input Image')
            plt.axis('off')
            
            # True mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask, cmap='gray')
            plt.title('True Mask')
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
            
            # Save figure
            output_path = str(Path(output_dir) / f'output_{i}.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.9),
        A.OneOf([
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(contrast_limit=0.2, p=1),
            A.HueSaturationValue(p=1),
        ], p=0.9),
        A.Lambda(mask=lambda x, **kwargs: x.round().clip(0, 1))
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    return A.Compose([
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.Resize(height=320, width=320, always_apply=True)
    ])

def get_preprocessing(preprocessing_fn):
    return A.Compose([A.Lambda(image=preprocessing_fn)])

def get_next_exp_dir(base_output_dir):
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    exp_num = 1
    while (base_dir / f'exp{exp_num}').exists():
        exp_num += 1
    return str(base_dir / f'exp{exp_num}')

def plot_metrics(history, output_dir):
    """Plot training and validation loss and IoU score in two subplots."""
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: IoU Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['iou_score'], 'b-', label='Training IoU')
    plt.plot(epochs, history.history['val_iou_score'], 'r-', label='Validation IoU')
    plt.title('Training and Validation IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_path = str(Path(output_dir) / 'metrics_plot.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    # Parameters
    data_dir = './dataset_smoke/'
    label_to_pixel = {'background': 0, 'smoke': 255}
    base_output_dir = './output'
    output_dir = get_next_exp_dir(base_output_dir)
    backbone = 'resnet50'
    batch_size = 8
    lr = 0.0001
    epochs = 40
    pretrained_weights = './weights/resnet50.h5'

    # Paths
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    x_train_dir = os.path.join(train_dir, 'images')
    y_train_dir = os.path.join(train_dir, 'masks')
    x_valid_dir = os.path.join(valid_dir, 'images')
    y_valid_dir = os.path.join(valid_dir, 'masks')
    x_test_dir = os.path.join(test_dir, 'images')
    y_test_dir = os.path.join(test_dir, 'masks')

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Preprocessing
    preprocess_input = sm.get_preprocessing(backbone)

    # Datasets
    train_dataset = Dataset(
        x_train_dir, y_train_dir, label_to_pixel,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input)
    )
    valid_dataset = Dataset(
        x_valid_dir, y_valid_dir, label_to_pixel,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input)
    )
    test_dataset = Dataset(
        x_test_dir, y_test_dir, label_to_pixel,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input)
    )

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model Training
    trainer = ModelTrainer(backbone, num_classes=2, lr=lr, epochs=epochs, output_dir=output_dir, pretrained_weights=pretrained_weights)
    history = trainer.train(train_dataloader, valid_dataloader)

    # Plot metrics
    plot_metrics(history, output_dir)

    # Load best weights for evaluation
    trainer.model.load_weights(str(Path(output_dir) / 'best_model.h5'))

    # Evaluate
    scores = trainer.evaluate(test_dataloader)
    print(f"Test scores: {scores}")

    # Save results
    saver = ResultSaver(label_to_pixel)
    saver.save_results(test_dataset, trainer, output_dir)

if __name__ == '__main__':
    main()