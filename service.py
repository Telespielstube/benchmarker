import torch, time

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from model import Model

# Class to handles all 
class Service():

    # Initializes the Service object. 
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.criterion = torch.nn.CrossEntropyLoss() #combines LogSoftmax and NLLLoss in one single class. Good for classification.
        self.optimizer_name = ''
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.training_data = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform), self.batch_size, shuffle=True)
        self.validation_data = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform), self.batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        self.model.to(self.device)
        self.training_loss = []
        self.validation_loss = []
        self.validation_accuracy = []  

    # Trains the model with the downloaded training dataset. 
    # @return    the elapsed time to train the model with the given hyperparameters set in the __init__().
    def training(self, optimizer, optimizer_name, learning_rate, epochs):
        print('Training in progress')
        self.model.train()
        self.optimizer_name = optimizer_name
        correct = 0
        total = 0
        for epoch in range(epochs):
            running_loss = 0
            for images, labels in self.training_data: 
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
                running_loss += loss.item() # the value of this tensor as a standard Python number
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()     
            self.training_loss.append(running_loss / len(self.training_data.dataset))
            print(f'Epoch {epoch} - Training loss: {running_loss / len(self.training_data.dataset):.10f}')       

    # Validates the trained model of the with the appropriate validation data set and displays loss and accuracy on the screen.
    def validate_cifar(self, epochs):
        print('Validating trained model.')
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        for run in range(epochs):
            with torch.no_grad():
                for images, labels in self.validation_data:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    val_loss += self.criterion(outputs, labels).item() 
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                self.validation_loss.append(val_loss / len(self.validation_data.dataset))     
        self.validation_accuracy.append((100.0 * correct / len(self.validation_data.dataset) / epochs)) 
        print(f'{self.optimizer_name} test loss: {val_loss / len(self.validation_data.dataset):.10f}, and test accuracy: {correct / epochs}/{len(self.validation_data.dataset)} ({(100.0 * correct / len(self.validation_data.dataset)) / epochs:.0f}%)')