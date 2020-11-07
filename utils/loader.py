import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision.transforms import transforms


class CalibrationData(dataset.Dataset):
    def __init__(self, folder, im_size=(128, 128)):
        super(CalibrationData, self).__init__()
        self.files = []
        for item in os.listdir(folder):
            self.files.append(os.path.join(folder, item))
        self.trans = get_calibration_trans(im_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.trans(img)


def get_calibration_trans(im_size=(128, 128)):
    trans = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return trans
