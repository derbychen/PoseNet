from datasets.KittiDset import KittiDset
import transforms3d.euler as txe
import transforms3d.quaternions as txq
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_dataloaders(dset_id, batch_size):

    img_path = "/home/hsuan/dataset/sequences/" + dset_id + "/image_0"
    txt_path = "/home/hsuan/dataset/poses/" + dset_id + ".txt"

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dset_train = KittiDset(img_path, txt_path, transform = transform)

    dataloader = DataLoader(dset_train, batch_size, shuffle=False, num_workers=2)
  
    return dataloader