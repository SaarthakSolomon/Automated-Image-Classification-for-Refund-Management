# Importing Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


# 1. Data Preprocessing and Augmentation
data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(150),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val' : transforms.Compose([
        transforms.Resize(150),
        transforms.CenterCrop(150),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 2. Load the Dataset
data_dir="D://Refund_Model//Data//train"
image_datasets={
    'train': datasets.ImageFolder(root=data_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=data_dir, transform= data_transforms['val'])
}
#DataLoader- Load the data in batches
dataloaders={
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False)
}

# 3. Load the Pre-trained VGG16 Model and Modify it for Our Use Case
model= models.vgg16(pretrained=True)

# Freeze earlier layers
for param in model.features.parameters():
    param.requires_grad = False
# Unfreeze last 4 layers of the feature extractor
for param in model.features[-4:]:  
    param.requires_grad = True

#Modify the classifier to match the number of the output classes
model.classifier[6]=nn.Sequential(
    nn.Linear(4096,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,22)
    
    
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is being used.")
# Move the Model to GPU if avaliable
model = model.to(device)


# 4. Define Loss Function and Optimizer
criterion= nn.CrossEntropyLoss()#Cross-entropy loss for multi-class classification
optimizer= optim.Adam(model.parameters(), lr=0.000001) #Adam optimizer with a low learning rate

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Step scheduler

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                self.early_stop = True

# 5. Training the Model
def train_model(model, dataloaders, criterion, optimizer, scheduler, early_stopping, num_epochs=15):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

        # Step the learning rate scheduler
        scheduler.step()
        
        # Check early stopping
        early_stopping(history['val_loss'][-1])
        if early_stopping.early_stop:
            print("Stopping training early")
            break

    return history

# Initialize EarlyStopping
early_stopping = EarlyStopping(patience=3)

# Call the training function with scheduler and early stopping
history = train_model(model, dataloaders, criterion, optimizer, scheduler, early_stopping, num_epochs=15)

# Now you can monitor the history and plot it later if needed.

# 6. Save the Trained Model
torch.save(model.state_dict(),'final_refund_item_classifier.pth')

# 7. Plot Training History
# Convert the tensors in history to CPU before plotting
train_acc_cpu = [acc.cpu().numpy() for acc in history['train_acc']]
val_acc_cpu = [acc.cpu().numpy() for acc in history['val_acc']]
train_loss_cpu = history['train_loss']  # Losses are typically on CPU, but you can verify
val_loss_cpu = history['val_loss']  # Same for validation loss

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_loss_cpu, label='Train Loss')
plt.plot(val_loss_cpu, label='Val Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_cpu, label='Train Accuracy')
plt.plot(val_acc_cpu,label ='Val Accuracy')
plt.title('Accuracy History')
plt.legend()
plt.show()

# Computing Metrics

# Function to make predictions and compute the metrics
def compute_metrics(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    # No gradient calculation needed during evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store the predictions and actual labels
            all_preds.extend(preds.cpu().numpy())  # Move tensors to CPU and convert to numpy
            all_labels.extend(labels.cpu().numpy())
    
    # Compute accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Print classification report for a detailed breakdown
    class_names = image_datasets['train'].classes  # List of class names
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\nClassification Report:\n", report)

# Call the function to compute and print metrics
compute_metrics(model, dataloaders['val'])
# Function to make predictions and compute the confusion matrix
def compute_confusion_matrix(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    # No gradient calculation needed during evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store the predictions and actual labels
            all_preds.extend(preds.cpu().numpy())  # Move tensors to CPU and convert to numpy
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(15,20))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Call the function to compute and plot the confusion matrix
class_names = image_datasets['train'].classes  # List of class names (40 classes)
cm = compute_confusion_matrix(model, dataloaders['val'])
plot_confusion_matrix(cm, class_names)