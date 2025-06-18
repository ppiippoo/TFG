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

def lpips_distance(loss_fn, tensor_img0, tensor_img1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_img0 = tensor_img0.to(device)
    tensor_img1 = tensor_img1.to(device)
    loss_fn = loss_fn.to(device)
    
    tensor_img0 = tensor_img0.unsqueeze(0) * 2 - 1
    tensor_img1 = tensor_img1.unsqueeze(0) * 2 - 1
    
    # Compute distance
    with torch.no_grad():
        dist01 = loss_fn.forward(tensor_img0, tensor_img1)
    
    return dist01.item() 

def get_datasets(data_dir, train_size=20000, test_size=2000):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    # Basic transform: resize/crop images and convert to tensor.
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.CenterCrop(224),
        transforms.ToTensor()  # Produces tensors in [0, 1]
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # If the dataset has more images than needed, take a random subset.
    if len(train_dataset) > train_size:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        train_dataset = Subset(train_dataset, indices[:train_size])
    if len(test_dataset) > test_size:
        indices = list(range(len(test_dataset)))
        random.shuffle(indices)
        test_dataset = Subset(test_dataset, indices[:test_size])

    return train_dataset, test_dataset

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


# Training and evaluation functions
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
        "color_enhancement": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        # Assuming the images are at least 224x224; adjust as necessary.
        "random_crop": transforms.Compose([transforms.RandomCrop((random.randint(height // 3, 3 * height // 4 ), random.randint(width // 3, 3 * width // 4))), transforms.Resize((height,width))]),
        "blur": transforms.GaussianBlur(kernel_size=(7, 13),sigma=(9, 11))
    }
    return transformations


# Experiment 1: Clean Training (no leakage)
def experiment1(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 1] Clean Training: No train/test leakage.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)


# Experiment 2: Direct Test Leakage – add every test image to train set (remove an equal number from train)
def experiment2(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 2] Direct Test Leakage: Inserting test images into training set.")
    num_to_remove = len(test_dataset)

    # Remove extra training images to free up space
    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    kept_indices = all_indices[num_to_remove:]
    new_train_examples = [train_dataset[i] for i in kept_indices]

    # Append all test images into training set.
    for sample in test_dataset:
        new_train_examples.append(sample)

    new_train_dataset = CustomListDataset(new_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    model = get_model( model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)

# Experiment 2/3: Put each test image to the train set 6 times 
def experiment23(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 2/3] Direct Test Leakage: Inserting test images into training set 6 times.")
    num_to_remove = len(test_dataset)*6

    # Remove extra training images to free up space
    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    kept_indices = all_indices[num_to_remove:]
    new_train_examples = [train_dataset[i] for i in kept_indices]

    # Append all test images into training set.
    for sample in test_dataset:
        for i in range(6):
            new_train_examples.append(sample)

    new_train_dataset = CustomListDataset(new_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    model = get_model( model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)



# Experiment 3: Augmented Transformation Leakage
# For each test image, generate 6 transformed versions and add them to the training set.
# The training set size is kept constant by removing a number of original training images.
def experiment3(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 3] Augmented Transformation Leakage: Adding transformed test images.")
    transformed_images = []
    # For each test image, create transformed versions
    for img, label in tqdm(test_dataset, desc="Transforming Test Images"):
        transformations = get_transformations(img)
        #for t_name in ["hflip", "vflip", "random_rotation", "random_crop", "color_enhancement","blur"]:
        #    transformed = transform_tensor_image(img,t_name)
        for t_name, transform in transformations.items():
            transformed = transform(img)
            #show_one_image(transformed)
            transformed_images.append((transformed, label))
    
    T = len(transformed_images)  # total new images from test
    num_to_remove = T

    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    kept_indices = all_indices[num_to_remove:]
    new_train_examples = [train_dataset[i] for i in kept_indices]

    # Add the transformed test images.
    new_train_examples.extend(transformed_images)
    new_train_dataset = CustomListDataset(new_train_examples)
    
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)



# Experiment 4: LPIPS Filtering
# Use LPIPS to evaluate similarities between test images and training images,
# and remove for each test image the 6 most similar training images.
def experiment4(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
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
    loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        loss_fn.cuda()
    else:
        loss_fn.cpu()

    # For each test image, compute LPIPS distance to every training sample and remove the 6 most similar.
    print("Filtering training images based on LPIPS distances...")
    indices_to_remove = set()
    # THIS LOOP IS EXPENSIVE
    for test_img, _ in tqdm(test_dataset, desc="LPIPS Filtering"):
        distances = []  # list of (idx, distance)
        for j, (train_img, _) in enumerate(new_train_examples):
            d = lpips_distance(loss_fn, train_img, test_img)
            distances.append((j, d))
        distances.sort(key=lambda x: x[1])
        # Remove 6 most similar training images for this test image.
        for k in range(6):
            indices_to_remove.add(distances[k][0])

    # Remove selected training images.
    filtered_train_examples = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove]

    # To keep the training set size consistent, we need to add extra samples.
    if len(filtered_train_examples) < len(new_train_examples):
        num_needed = len(new_train_examples) - len(filtered_train_examples)
        if len(filtered_train_examples) > 0:
            #additional_examples = random.sample(filtered_train_examples, num_needed)
            additional_examples = random.sample(train_examples_removed, num_needed)
            filtered_train_examples.extend(additional_examples)
    
    new_train_dataset = CustomListDataset(filtered_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)


# Experiment 5: Transformation Impact Analysis
# For each transformation, add the transformed test image to training set,
# then for each test image compute the LPIPS distance to the 3 most similar training samples.
# The average of these distances across test images is calculated per transformation.

def experiment5_helper(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate, loss_fn, t_name): 
    # Create a training set solely from test images transformed with t_name.
    transformed_images = []
    for img, label in test_dataset:
        transform = get_transformations(img).get(t_name)
        transformed_images.append((transform(img),label))

    T = len(transformed_images)
    num_to_remove = T

    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    kept_indices = all_indices[num_to_remove:]
    new_train_examples = [train_dataset[i] for i in kept_indices]    
    new_train_examples.extend(transformed_images)
    
    distances_per_test = []
    
    # For each test image, compute LPIPS to every image in the current transformed training set.
    for test_img, _ in tqdm(test_dataset, desc=f"LPIPS for {t_name}"):
        dists = []
        for train_img, _ in new_train_examples:
            #test_tensor = test_img.unsqueeze(0) * 2 - 1
            #train_tensor = train_img.unsqueeze(0) * 2 - 1
            #with torch.no_grad():
            #    d = loss_fn(test_tensor, train_tensor).item()
            d = lpips_distance(loss_fn, train_img, test_img)
            dists.append(d)
        # Find the 3 most similar images
        dists.sort()
        top3 = dists[:3]
        mean_dist = sum(top3) / 3
        distances_per_test.append(mean_dist)
    overall_mean = sum(distances_per_test) / len(distances_per_test)
    print(f"Transformation: {t_name}, Mean LPIPS distance: {overall_mean:.4f}")

    
    new_train_dataset = CustomListDataset(transformed_train)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #num_classes = len(train_dataset.dataset.classes) if isinstance(train_dataset, Subset) else len(train_dataset.classes)
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate)
    
    return overall_mean, accuracy
    

def experiment5(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate):
    print("[Experiment 5] Transformation Impact Analysis")
    transformations = get_transformations(train_dataset[0][0])
    results = {}
    
    loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        loss_fn.cuda()
    else:
        loss_fn.cpu()
        
    # For each transformation check its impact separately:
    for t_name, transform in transformations.items():
        print(f"\nProcessing transformation: {t_name}")
        overall_mean, accuracy = experiment5_helper(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate, loss_fn, t_name)
        results[t_name] = (overall_mean, accuracy)
       
    print("\nOverall Transformation Impact Results:")
    for t_name, (score,accuracy) in results.items():
        print(f"{t_name}: mean={score:.4f} accuracy={accuracy:.2f}")


def main():
    if(len(sys.argv) < 3):
        print("wrong usage: python3 experiments.py <1/2/3/4/5> <alexnet/resnet50>")
        exit(-1)

    data_dir = "./cifar_images_train_test/CIFAR-10-images/"
    experiment = int(sys.argv[1])        # 1,2,3,4,5,
    model = sys.argv[2]     # "alexnet" or "resnet50"
    batch_size = 64
    num_classes = 10
    epochs =  20
    learning_rate = 0.0001

    train_dataset, test_dataset = get_datasets(data_dir)
    if experiment == 1:
        experiment1(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate)
    elif experiment == 2:
        experiment2(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate)
    elif experiment == 23:
        experiment23(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate)
    elif experiment == 3:
        experiment3(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate)
    elif experiment == 4:
        experiment4(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate)
    elif experiment == 5:
        experiment5(train_dataset, test_dataset, model, batch_size, num_classes, epochs, learning_rate)

if __name__ == "__main__":
    main()
