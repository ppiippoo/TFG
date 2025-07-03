import os
import random
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset

from torchvision import transforms, datasets, models
import lpips  # pip install lpips

# **********************
# TO SET THE RANDOM SEED
# **********************
random.seed(10)


def lpips_distance(loss_fn, tensor_img0, tensor_img1):
    
    tensor_img0 = tensor_img0.unsqueeze(0) * 2 - 1
    tensor_img1 = tensor_img1.unsqueeze(0) * 2 - 1
    
    # Compute distance
    with torch.no_grad():
        dist01 = loss_fn.forward(tensor_img0, tensor_img1)
    
    return dist01.item() 

def get_datasets(data_dir, train_size=20000, test_size=2000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Required because LPIPS models (like AlexNet, VGG) expect larger images
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # If the dataset has more images than needed, take a random subset.
    if len(train_dataset) > train_size:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        new_train_dataset = Subset(train_dataset, indices[:train_size])
        remaining_train_dataset = Subset(train_dataset, indices[train_size:])
    if len(test_dataset) > test_size:
        indices = list(range(len(test_dataset)))
        random.shuffle(indices)
        test_dataset = Subset(test_dataset, indices[:test_size])

    return new_train_dataset, remaining_train_dataset, test_dataset

# Helper Dataset to hold a list of image samples
class CustomListDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: list of (image_tensor, label) tuples
        """
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def get_model(model_arch="alexnet", num_classes=10):
    if model_arch.lower() == "alexnet":
        #model = models.alexnet(pretrained=True)
        model = models.alexnet(weights=None)
        # Replace the classifier last layer.
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        #model = models.resnet50(pretrained=True)
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    final_accuracy = 0
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = 0.005, momentum = 0.9)  

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        final_accuracy = test_model(model, test_loader, device)
    return final_accuracy
        
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return (100 * correct / total)

def get_transformations(image_tensor):
    _, height, width = image_tensor.shape
    transformations = {
        "horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),
        "vertical_flip": transforms.RandomVerticalFlip(p=1.0),
        "rotation": transforms.RandomRotation(degrees=random.randint(30,300)),
        "color_enhancement": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        # Assuming the images are at least 224x224; adjust as necessary.
        "random_crop": transforms.Compose([transforms.RandomCrop((random.randint(height // 3, 3 * height // 4 ), random.randint(width // 3, 3 * width // 4))), transforms.Resize((height,width))]),
        "blur": transforms.GaussianBlur(kernel_size=(7, 25),sigma=(9, 11))
    }
    return transformations

def repopulate(filtered_train_examples, new_train_examples, train_examples_removed, excluded_train_dataset):
    if len(filtered_train_examples) < len(new_train_examples):
        num_needed = len(new_train_examples) - len(filtered_train_examples)
        print("Number of images removed: ", num_needed)
        if len(train_examples_removed) >= num_needed:
            #additional_examples = random.sample(filtered_train_examples, num_needed)
            additional_examples = random.sample(train_examples_removed, num_needed)
            filtered_train_examples.extend(additional_examples)
        elif len(excluded_train_dataset) >= num_needed:
            excluded_train_indix = list(range(len(excluded_train_dataset)))
            excluded_train_examples = [excluded_train_dataset[i] for i in excluded_train_indix[:num_needed]]
            
            additional_examples = random.sample(excluded_train_examples, num_needed)
            filtered_train_examples.extend(additional_examples)
        elif len(train_examples_removed) + len(excluded_train_dataset) >= num_needed:
            additional_examples = random.sample(train_examples_removed, len(train_examples_removed))
            filtered_train_examples.extend(additional_examples)
            
            excluded_train_indix = list(range(len(excluded_train_dataset)))
            excluded_train_examples = [excluded_train_dataset[i] for i in excluded_train_indix[:(num_needed-len(train_examples_removed))]]

            
            additional_examples = random.sample(excluded_train_examples, num_needed-len(train_examples_removed))
            filtered_train_examples.extend(additional_examples)
        else:
            print("too many examples removed")
            exit(-1)
    return filtered_train_examples

# Experiment 4: LPIPS Filtering
# Use LPIPS to evaluate similarities between test images and training images,
# and remove for each test image the 6 most similar training images.
def experiment4(train_dataset, excluded_train_dataset, test_dataset, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 4] LPIPS Filtering: Removing similar training images based on LPIPS.")
    # First, generate the training set with transformed test images, as in experiment3.
    transformed_images = []
    for img, label in tqdm(test_dataset, desc="Transforming Test Images"):
        transformations = get_transformations(img)
        for t_name, transform in transformations.items():
            transformed = transform(img)
            transformed_images.append((transformed, label))
    
    T = len(transformed_images)
    num_to_remove = T

    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    kept_indices = all_indices[num_to_remove:]
    remv_indices = all_indices[:num_to_remove]
    new_train_examples = [train_dataset[i] for i in kept_indices]
    train_examples_removed = [train_dataset[i] for i in remv_indices] 
    
    new_train_examples.extend(transformed_images)

    # Initialization of LPIPS loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to(device)

    # For each test image, compute LPIPS distance to every training sample and remove the 6 most similar.
    print("Filtering training images based on LPIPS distances...")
    neigh_to_remove = 6
    print("Removing for each test image ", neigh_to_remove)
    indices_to_remove_1 = set()
    indices_to_remove_2 = set()
    indices_to_remove_3 = set()
    indices_to_remove_4 = set()

    # THIS LOOP IS EXPENSIVE
    #for test_img, _ in tqdm(test_dataset, desc="LPIPS Filtering"):
    counting = 0
    len_test_dataset = len(test_dataset)
    for test_img, _ in test_dataset:
        print("LPIPS filtering: ", counting, "/", len_test_dataset, end="\r", flush=True)
        counting +=1
        
        distances = []  # list of (idx, distance)
        tensor_img0 = test_img.to(device)
        
        for j, (train_img, _) in enumerate(new_train_examples):
            tensor_img1 = train_img.to(device)
            
            d = lpips_distance(loss_fn, tensor_img0, tensor_img1)
            distances.append((j, d))
            
        distances.sort(key=lambda x: x[1])
        
        # Remove 6 most similar training images for this test image.
        for k in range(neigh_to_remove):
            indices_to_remove_1.add(distances[k][0])
        # Remove 10 most similar training images for this test image.
        for k in range(10):
            indices_to_remove_2.add(distances[k][0])
        # Remove 12 most similar training images for this test image.
        for k in range(12):
            indices_to_remove_3.add(distances[k][0])
        # Remove 15 most similar training images for this test image.
        for k in range(15):
            indices_to_remove_4.add(distances[k][0])

    # Remove selected training images.
    filtered_train_examples_1 = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove_1]
    filtered_train_examples_2 = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove_2]
    filtered_train_examples_3 = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove_3]
    filtered_train_examples_4 = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove_4]



    # To keep the training set size consistent, we need to add extra samples.
    repopulate(filtered_train_examples_1, new_train_examples, train_examples_removed, excluded_train_dataset)
    repopulate(filtered_train_examples_2, new_train_examples, train_examples_removed, excluded_train_dataset)
    repopulate(filtered_train_examples_3, new_train_examples, train_examples_removed, excluded_train_dataset)
    repopulate(filtered_train_examples_4, new_train_examples, train_examples_removed, excluded_train_dataset)

    new_train_dataset = CustomListDataset(filtered_train_examples_1)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    
    # training alexnet and resnet removing 6 images per test item
    print("Num neigh: 6")
    print("Training ALEXNET")
    model = get_model("alexnet", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_alexnet_1 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    print("")
    print("Training RESNET-50")
    model = get_model("resnet50", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_resnet_1 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    # training alexnet and resnet removing 10 images per test item
    new_train_dataset = CustomListDataset(filtered_train_examples_2)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("")
    print("Num neigh: 10")
    print("Training ALEXNET")
    model = get_model("alexnet", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_alexnet_2 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    print("")
    print("Training RESNET-50")
    model = get_model("resnet50", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_resnet_2 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    # training alexnet and resnet removing 12 images per test item
    new_train_dataset = CustomListDataset(filtered_train_examples_3)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("")
    print("Num neigh: 12")
    print("Training ALEXNET")
    model = get_model("alexnet", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_alexnet_3 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    print("")
    print("Training RESNET-50")
    model = get_model("resnet50", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_resnet_3 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    # training alexnet and resnet removing 15 images per test item
    new_train_dataset = CustomListDataset(filtered_train_examples_4)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("")
    print("Num neigh: 15")
    print("Training ALEXNET")
    model = get_model("alexnet", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_alexnet_4 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    print("")
    print("Training RESNET-50")
    model = get_model("resnet50", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_resnet_4 = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    print(f"Accuracy removing 6: alexnet={accuracy_alexnet_1:.2f} accuracy_resnet={accuracy_resnet_1:.2f}")
    print(f"Accuracy removing 10: alexnet={accuracy_alexnet_2:.2f} accuracy_resnet={accuracy_resnet_2:.2f}")
    print(f"Accuracy removing 12: alexnet={accuracy_alexnet_3:.2f} accuracy_resnet={accuracy_resnet_3:.2f}")
    print(f"Accuracy removing 6: alexnet={accuracy_alexnet_4:.2f} accuracy_resnet={accuracy_resnet_4:.2f}")



def experiment4_threshold(train_dataset, excluded_train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 4] LPIPS Filtering with threshold: Removing similar training images based on LPIPS.")
    # First, generate the training set with transformed test images, as in experiment3.
    transformed_images = []
    for img, label in tqdm(test_dataset, desc="Transforming Test Images"):
        transformations = get_transformations(img)
        for t_name, transform in transformations.items():
            transformed = transform(img)
            transformed_images.append((transformed, label))
    
    T = len(transformed_images)
    num_to_remove = T

    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    kept_indices = all_indices[num_to_remove:]
    remv_indices = all_indices[:num_to_remove]
    new_train_examples = [train_dataset[i] for i in kept_indices]
    
    excluded_indices = list(range(len(excluded_train_dataset)))
    excluded_train_examples = [excluded_train_dataset[i] for i in excluded_indices]
    
    train_examples_removed = [train_dataset[i] for i in remv_indices] 
    
    new_train_examples.extend(transformed_images)

    # Initialization of LPIPS loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to(device)

    threshold = 0.5
    # For each test image, compute LPIPS distance to every training sample and remove the 6 most similar.
    print("Filtering training images based on LPIPS distances...")
    indices_to_remove = set()
    # THIS LOOP IS EXPENSIVE
    for test_img, _ in tqdm(test_dataset, desc="LPIPS Filtering"):
        tensor_img0 = test_img.to(device)
        for j, (train_img, _) in enumerate(new_train_examples):
            tensor_img1 = train_img.to(device)
            
            d = lpips_distance(loss_fn, tensor_img0, tensor_img1)
            if d <= threshold:
                indices_to_remove.add(j)
    # Remove selected training images.
    filtered_train_examples = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove]

    repopulate(filtered_train_examples, new_train_examples, train_examples_removed,excluded_train_dataset)
    
    new_train_dataset = CustomListDataset(filtered_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    print("Training ALEXNET")
    model = get_model("alexnet", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

    print("")
    print("Training RESNET-50")
    model = get_model("resnet50", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)


def main():
    if(len(sys.argv) < 2):
        print("wrong usage: python3 experiments.py <1/2>")
        exit(-1)

    data_dir = "./cifar_images_train_test/CIFAR-10-images/"
    experiment = int(sys.argv[1])        # 1,2
    batch_size = 64
    num_classes = 10
    epochs =  20
    learning_rate = 0.0001

    train_dataset, excluded_train_dataset, test_dataset = get_datasets(data_dir)
    if experiment == 1:
        experiment4(train_dataset, excluded_train_dataset, test_dataset, batch_size, num_classes, epochs, learning_rate)
    elif experiment == 2:
        experiment4_threshold(train_dataset, excluded_train_dataset, test_dataset, batch_size, num_classes, epochs, learning_rate)
        
if __name__ == "__main__":
    main()
