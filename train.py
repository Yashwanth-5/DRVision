import torch

def train_model(model, train_loader, criterion, optimizer, device, epochs=10): [cite: 682, 683]
    model.to(device) [cite: 684]
    for epoch in range(epochs): [cite: 685]
        model.train() [cite: 686]
        running_loss = 0.0 [cite: 687]
        
        for images, labels in train_loader: [cite: 688]
            images = images.to(device) [cite: 689, 692]
            labels = labels.to(device) [cite: 690, 693]
            
            outputs = model(images) [cite: 691, 694]
            loss = criterion(outputs, labels) [cite: 695]
            
            optimizer.zero_grad() [cite: 696]
            loss.backward() [cite: 697]
            optimizer.step() [cite: 698]
            
            running_loss += loss.item() [cite: 699]
            
        epoch_loss = running_loss / len(train_loader) [cite: 700, 701]
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}") [cite: 702]
