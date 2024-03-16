import os, sys
sys.path.append('/home/srinivi/crcv/xray_classification')
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from base import BaseDataLoader


class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, training=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        root_list = ['train', 'val'] if training else ['test']

        for root in root_list:
            for label, subdir in enumerate(['NORMAL', 'PNEUMONIA']):
                class_dir = os.path.join(root_dir, root, subdir)
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

class XRayDataloader(BaseDataLoader):    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, transform=None, training=True):
        self.data_dir = data_dir
        self.dataset = ChestXrayDataset(data_dir, transform=transform, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

if __name__ == '__main__':
    data_dir = '/home/srinivi/crcv/xray_classification/dataset/chest_xray/'
    trsfm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    train_loader = XRayDataloader(data_dir, 1, shuffle=True, validation_split=0.1, num_workers=1, transform=trsfm, training=True)
    valid_loader = train_loader.split_validation()
    print(len(train_loader))
    print(len(valid_loader))
