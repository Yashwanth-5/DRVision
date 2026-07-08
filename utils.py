import torch
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score [cite: 589]

# Label class mappings
class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"] [cite: 761, 762, 763, 764, 765, 767]

def evaluate_model(model, test_loader, device): [cite: 707]
    model.eval() [cite: 708]
    all_predictions = [] [cite: 709]
    all_labels = [] [cite: 709]
    
    with torch.no_grad(): [cite: 713]
        for images, labels in test_loader: [cite: 714]
            images = images.to(device) [cite: 714]
            outputs = model(images) [cite: 716]
            _, predictions = torch.max(outputs, 1) [cite: 717]
            all_predictions.extend(predictions.cpu().numpy()) [cite: 718]
            all_labels.extend(labels.numpy()) [cite: 719]
            
    accuracy = accuracy_score(all_labels, all_predictions) [cite: 715, 720]
    precision = precision_score(all_labels, all_predictions, average='weighted') [cite: 720]
    recall = recall_score(all_labels, all_predictions, average='weighted') [cite: 721, 722]
    f1 = f1_score(all_labels, all_predictions, average='weighted') [cite: 723]
    qwk = cohen_kappa_score(all_labels, all_predictions, weights='quadratic') [cite: 723]
    
    print("Accuracy:", accuracy) [cite: 724]
    print("Precision:", precision) [cite: 725]
    print("Recall:", recall) [cite: 726]
    print("F1 Score:", f1) [cite: 727]
    print("QWK:", qwk) [cite: 728]
    return accuracy, precision, recall, f1, qwk [cite: 729]

def generate_gradcam(model, image_tensor, target_layer, class_index): [cite: 733]
    model.eval() [cite: 733]
    gradients = [] [cite: 734]
    activations = [] [cite: 735]
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0]) [cite: 736]
    def forward_hook(module, input, output):
        activations.append(output) [cite: 737]
        
    target_layer.register_forward_hook(forward_hook) [cite: 738]
    target_layer.register_backward_hook(backward_hook) [cite: 738]
    
    output = model(image_tensor) [cite: 741]
    score = output[:, class_index] [cite: 742, 743]
    model.zero_grad() [cite: 744]
    score.backward() [cite: 745]
    
    grad = gradients[0] [cite: 746]
    act = activations[0] [cite: 747]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True) [cite: 748]
    
    cam = torch.sum(weights * act, dim=1) [cite: 749, 751]
    cam = torch.relu(cam) [cite: 752]
    cam = cam.squeeze().detach().cpu().numpy() [cite: 753]
    cam = cv2.resize(cam, (512, 512)) [cite: 754]
    cam = (cam - cam.min()) / (cam.max() - cam.min()) [cite: 755, 756]
    return cam [cite: 757]

def predict_image(model, image_path, transform, device): [cite: 769]
    image = Image.open(image_path).convert("RGB") [cite: 770, 772]
    image = transform(image).unsqueeze(0).to(device) [cite: 771, 773]
    model.eval() [cite: 774]
    
    with torch.no_grad(): [cite: 775]
        output = model(image) [cite: 776]
        probabilities = torch.softmax(output, dim=1) [cite: 777]
        confidence, predicted_class = torch.max(probabilities, 1) [cite: 778]
        predicted_label = class_names[predicted_class.item()] [cite: 779]
        
    return predicted_label, confidence.item() [cite: 782]
