import os
import csv
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

############################################
# Step 1: Load Data and Manually Encode Labels
X = []
y = []

#reads each row (first 1404 values, label (last))
with open(r'C:\face-emoji\data\landmarks_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        *features, label = row
        X.append([float(x) for x in features])
        y.append(label)

#converts features to float and stores the label
X = np.array(X, dtype=np.float32)

# Create a manual mapping for label encoding
unique_labels = sorted(list(set(y)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
print("Label mapping:", label_to_index)

# Convert labels to integers
y_encoded = [label_to_index[label] for label in y]
y_encoded = np.array(y_encoded, dtype=np.int64)

############################################
# Step 2: Manually Split Data into Train and Test Sets

# Combine X and y for shuffling
combined = list(zip(X, y_encoded)) #pairs each landmark vector (X) with label (y)
random.shuffle(combined) #mixes order so that training set isnt biased by label ordering

split_idx = int(len(combined) * 0.8) #gets index at 80% of the dataset length
train_data = combined[:split_idx] #first 80%
test_data = combined[split_idx:] #last 20%

X_train, y_train = zip(*train_data) #X_train: Landmark vectors, y_train: Encoded labels
X_test, y_test = zip(*test_data)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32) #torch.tensor() converts lists into pytorch tensors
y_train = torch.tensor(np.array(y_train), dtype=torch.long)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_test = torch.tensor(np.array(y_test), dtype=torch.long)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

############################################
# Step 3: Define the Model
class EmotionClassifier(nn.Module):
    def __init__(self, input_size=1404, hidden1=512, hidden2=256, hidden3=128, output_size=7):
        super(EmotionClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden3, output_size)
        )

    def forward(self, x):
        return self.network(x)

# The number of classes is determined by the number of unique labels
num_classes = len(unique_labels)
model = EmotionClassifier(input_size=1404, hidden1=512, hidden2=256, hidden3=128, output_size=num_classes)
# model = EmotionClassifier(input_size=1404, hidden_size=256, output_size=num_classes)
print(model)

############################################
# Step 4: Train the Model

def train_model(model, X, y, n_epochs=50):
    loss_fn = nn.CrossEntropyLoss() #used for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adam: advanced version of SGD that adapts to learning rate
    #model.parameters(): tells Pytorch which weights to update
    losses = []

    for epoch in range(n_epochs):
        model.train() # puts model in training mode (activates dropout/batch norm if used)
        outputs = model(X) #performs forward pass--generates predicted scores (logits)
        loss = loss_fn(outputs, y) # compares model predictions vs actual labels

        optimizer.zero_grad() #resets previous gradients (important for every iteration)
        loss.backward() # performs backpropagation (calculates gradients of the loss w.r.t. model weights)
        optimizer.step() #updates the weights using calculated gradients

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            losses.append(loss.item())
    return losses

losses = train_model(model, X_train, y_train, n_epochs=50)

# Save the model
os.makedirs(r'C:\face-emoji\models', exist_ok=True)
torch.save(model.state_dict(), r'C:\face-emoji\models\emotion_model.pth')
print("Model saved to C:\\face-emoji\\models\\emotion_model.pth")

############################################
# Step 5: Evaluate the Model

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_test).float().mean().item()
    print(f"Test accuracy: {accuracy:.4f}")

############################################
# Step 6: Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(range(0, len(losses) * 5, 5), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(alpha=0.3)
plt.savefig('training_loss_emotion_model.png')
plt.close()
