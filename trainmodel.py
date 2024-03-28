import torch
from torchvision import transforms, datasets, models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

root_dir = 'E:/Magic the Gathering/Botv3/MTG-Machine-Learning-main/MTG-Machine-Learning-main/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((336, 468)), transforms.ToTensor()])
train_data = datasets.ImageFolder(root_dir + 'training', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_data = datasets.ImageFolder(root_dir + 'validation', transform=transform)
validation_loader = DataLoader(validation_data, batch_size=32)

model = densenet121(weights=DenseNet121_Weights.DEFAULT)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(train_data.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(11):
    model.train()
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/11 - Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), f'{root_dir}model_checkpoint_epoch_{epoch+1}.pth')

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in tqdm(validation_loader, desc=f'Epoch {epoch+1}/11 - Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: Validation Accuracy: {accuracy:.2f}%')
