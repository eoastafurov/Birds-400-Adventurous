import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from tqdm import tqdm
import sys

# sys.path.append('../')
from data import BirdsDataManger
from transforms import AlbuTransformCollection


def get_mean_and_std():
    dm = BirdsDataManger(
        root='./dataset/',
        batch_size=256,
        img_size=224,
        # grayscale=False,
        grayscale=True,
        transforms_collection=AlbuTransformCollection,
        _supervise_whole_dataset=True
    )
    dm.prepare_data()
    dm.setup()
    dataloader = dm.supervise_dataloder()

    ####
    device = torch.device('cuda:0')
    ###

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    with torch.no_grad():
        for data, _ in tqdm(dataloader):
            # Mean over batch, height and width, but not over the channels
            data.to(device)
            channels_sum += torch.mean(data, dim=[0,2,3]).detach().cpu()
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3]).detach().cpu()
            num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__':
    print(get_mean_and_std()) 
    # COLOR: (tensor([0.4497, 0.4503, 0.4181]), tensor([0.2445, 0.2435, 0.2646]))
    # BW: (tensor([0.4468, 0.4468, 0.4468]), tensor([0.2408, 0.2408, 0.2408]))