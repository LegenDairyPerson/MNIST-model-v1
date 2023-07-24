import torch
import torchvision
import matplotlib
import model
from PIL import Image

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)
test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
def imageToTensor(image):
    startImage = Image.open(image, mode="r")
    transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])

    tensor = transform(startImage)
    tensor = tensor / 255
    return tensor

tensor = imageToTensor("test sample 1.png")

input = (tensor, 1)
#print(vars(test_loader))
newNetwork = model.Net()
model_dict = torch.load("model.pth")
newNetwork.load_state_dict(model_dict)
sample_idx = torch.randint(len(test_data), size=(1,)).item()

for i in range (1, 5):
    sample_idx = torch.randint(len(test_data), size=(1,)).item()

model.testsimple(newNetwork, input)
