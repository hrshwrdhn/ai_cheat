# downloading the dataset and creating PyTorch datasets to load the data

```
from torchvision.datasets.utils import download_url
import tarfile

# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, './cifer')

# Extract from archive file
with tarfile.open('./cifer/cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')


# Dowload the dataset
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_url(dataset_url, '.')

# Extract from archive
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')
    
# Look into the data directory
data_dir = './data/cifar10'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)

```
