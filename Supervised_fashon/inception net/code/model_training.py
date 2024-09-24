# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, accuracy_score
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from config import device, num_classes, num_epochs
from Model import Model
from tqdm import tqdm
from Util import make_mask
from sklearn.metrics import precision_score, recall_score, accuracy_score




def initialize_model():
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)


    # Define the nodes up to which we want the features (excluding the final fc layer)
    return_nodes = {
        'classifier.dropout': 'features'
    }

    # Create the feature extractor
    feature_extractor = create_feature_extractor(model, return_nodes)

    model = Model(feature_extractor, num_classes=num_classes)
    model = model.to(device)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion




def train_model(model, optimizer, criterion, train_dataloader, validation_dataloader):
    train_losses = []
    validation_losses = []
    validation_accuracies = []
    validation_precisions = []
    validation_recalls = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        epoch_validation_loss = 0.0
        epoch_validation_correct = 0
        epoch_validation_total = 0

        if epoch == int(num_epochs / 3):
            for name, param in model.feature_extractor.named_parameters():
                if name.startswith('layers.6'):
                    param.requires_grad = True

        if epoch == 2 * int(num_epochs / 3):
            for name, param in model.feature_extractor.named_parameters():
                param.requires_grad = True

        # Training phase
        model.train()
        for batch_inputs, batch_anatomical_location, batch_targets in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            mask = make_mask(batch_anatomical_location, num_classes)
            outputs = model(batch_inputs, mask)

            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            epoch_train_total += batch_targets.size(0)
            epoch_train_correct += (predicted == batch_targets).sum().item()

        train_accuracy = 100 * epoch_train_correct / epoch_train_total
        train_losses.append(epoch_train_loss / len(train_dataloader))

        # validationing phase
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for batch_inputs, batch_anatomical_location, batch_targets in tqdm(validation_dataloader, desc=f'validationing Epoch {epoch+1}/{num_epochs}'):
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

                mask = make_mask(batch_anatomical_location, num_classes)
                
                outputs = model(batch_inputs, mask)

                loss = criterion(outputs, batch_targets)
                epoch_validation_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                epoch_validation_total += batch_targets.size(0)
                epoch_validation_correct += (predicted == batch_targets).sum().item()

                all_targets.extend(batch_targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        validation_accuracy = 100 * epoch_validation_correct / epoch_validation_total
        validation_precision = precision_score(all_targets, all_predictions, average='weighted')
        validation_recall = recall_score(all_targets, all_predictions, average='weighted')

        validation_losses.append(epoch_validation_loss / len(validation_dataloader))
        validation_accuracies.append(validation_accuracy)
        validation_precisions.append(validation_precision)
        validation_recalls.append(validation_recall)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%, validation Loss: {validation_losses[-1]:.4f}, validation Accuracy: {validation_accuracy:.2f}%, validation Precision: {validation_precision:.4f}, validation Recall: {validation_recall:.4f}')

    return train_losses, validation_losses, validation_accuracies, validation_precisions, validation_recalls
