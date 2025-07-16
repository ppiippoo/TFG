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

# global variables
random.seed(10)
batch_size = 64
num_classes = 10
epochs =  20
learning_rate = 0.0001

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


# Training and evaluation functions
def train_model(model, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
        "random_crop": transforms.Compose([transforms.RandomCrop((random.randint(height // 3, 3 * height // 4 ), random.randint(width // 3, 3 * width // 4))), transforms.Resize((height,width))]),
        "blur": transforms.GaussianBlur(kernel_size=(7, 25),sigma=(9, 11))
    }
    return transformations


# Experiment Clean Training (no leakage)
def clean_training(train_dataset, test_dataset, model):
    print("[Experiment 1] Clean Training: No train/test leakage.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device)

# Experiment 1.1: Clean training with calculation of the mean lpips distance (like in experiment 5) 
def clean_training_with_mean_lpips_distance(train_dataset, test_dataset, model):
    print("[Experiment 1.1] Clean Training: No train/test leakage.")
    distances_per_test = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to(device)
    
    # For each test image, compute LPIPS to every image in the current transformed training set.
    for test_img, _ in tqdm(test_dataset, desc=f"LPIPS for Clean train-test set"):
        dists = []
        tensor_img0 = test_img.to(device)
        for train_img, _ in train_dataset:
            tensor_img1 = train_img.to(device)
            
            d = lpips_distance(loss_fn, tensor_img0, tensor_img1)
            dists.append(d)
        # Find the 3 most similar images
        dists.sort()
        top3 = dists[:3]
        mean_dist = sum(top3) / 3
        distances_per_test.append(mean_dist)
    overall_mean = sum(distances_per_test) / len(distances_per_test)
    print(f"Mean LPIPS distance: {overall_mean:.4f}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device)

# Experiment Exact duplicates Contamination – add every test image to train set (remove an equal number from train)
def exact_duplicates_contamination(train_dataset, test_dataset, model):
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
    train_model(model, train_loader, test_loader, device)

# Experiment: Put each test image to the train set 6 times 
def exct_duplicates_with_repetition_contamination(train_dataset, test_dataset, model):
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
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device)


# Experiment Near duplicate Contamination
# For each test image, generate 6 transformed versions and add them to the training set.
# The training set size is kept constant by removing a number of original training images.
def near_duplicates_contamination(train_dataset, test_dataset, model):
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
    train_model(model, train_loader, test_loader, device)



# Experiment Decontamination removing n near-duplicates
# Use LPIPS to evaluate similarities between test images and training images,
# and remove for each test image the 6 most similar training images.
def decontamination_removing_n_duplicates(train_dataset, excluded_train_dataset,test_dataset, model,neigh_to_remove = 6):
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
    print("Removing for each test image ", neigh_to_remove)
    indices_to_remove = set()
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
            indices_to_remove.add(distances[k][0])

    # Remove selected training images.
    filtered_train_examples = [ex for idx, ex in enumerate(new_train_examples) if idx not in indices_to_remove]

    # To keep the training set size consistent, we need to add extra samples.
    if len(filtered_train_examples) < len(new_train_examples):
        num_needed = len(new_train_examples) - len(filtered_train_examples)
        print("Number of images removed: ", num_needed)
        if len(train_examples_removed) >= num_needed:
            #additional_examples = random.sample(filtered_train_examples, num_needed)
            additional_examples = random.sample(train_examples_removed, num_needed)
            filtered_train_examples.extend(additional_examples)
        elif len(excluded_train_dataset) >= num_needed:
            excluded_train_examples = list(excluded_train_dataset)
            additional_examples = random.sample(excluded_train_examples, num_needed)
            filtered_train_examples.extend(additional_examples)
        elif len(train_examples_removed) + len(excluded_train_dataset) >= num_needed:
            excluded_train_examples = list(excluded_train_dataset)
            additional_examples = random.sample(train_examples_removed, len(train_examples_removed))
            filtered_train_examples.extend(additional_examples)
            additional_examples = random.sample(excluded_train_examples, num_needed-len(train_examples_removed))
            filtered_train_examples.extend(additional_examples)
        else:
            print("too many examples removed")
            exit(-1)
    print("Size filtered train set after repopulation: ", len(filtered_train_examples))
    
    new_train_dataset = CustomListDataset(filtered_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device)


# Experiment Decontamination using threshold
# Use LPIPS to evaluate similarities between test images and training images,
# and remove for each test image the training images that have a distance less than the threshold.
def decontamination_with_threshold(train_dataset, excluded_train_dataset, test_dataset, model, threshold):
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

    # To keep the training set size consistent, we need to add extra samples.
    if len(filtered_train_examples) < len(new_train_examples):
        num_needed = len(new_train_examples) - len(filtered_train_examples)
        print("Number of images removed: ", num_needed)
        if len(train_examples_removed) >= num_needed:
            #additional_examples = random.sample(filtered_train_examples, num_needed)
            additional_examples = random.sample(train_examples_removed, num_needed)
            filtered_train_examples.extend(additional_examples)
        elif len(excluded_train_dataset) >= num_needed:
            excluded_train_examples = list(excluded_train_dataset)
            additional_examples = random.sample(excluded_train_examples, num_needed)
            filtered_train_examples.extend(additional_examples)
        elif len(train_examples_removed) + len(excluded_train_dataset) >= num_needed:
            excluded_train_examples = list(excluded_train_dataset)
            additional_examples = random.sample(train_examples_removed, len(train_examples_removed))
            filtered_train_examples.extend(additional_examples)
            additional_examples = random.sample(excluded_train_examples, num_needed-len(train_examples_removed))
            filtered_train_examples.extend(additional_examples)
        else:
            exit(-1)
    print("Size filtered train set after repopulation: ", len(filtered_train_examples))
    
    new_train_dataset = CustomListDataset(filtered_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, device)


# helper for experiment contamination_with_specific_transformation
def contamination_with_specific_transformation_helper(train_dataset, test_dataset, model, t_name): 
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn = loss_fn.to(device)
    
    # For each test image, compute LPIPS to every image in the current transformed training set.
    for test_img, _ in tqdm(test_dataset, desc=f"LPIPS for {t_name}"):
        dists = []
        tensor_img0 = test_img.to(device)
        for train_img, _ in new_train_examples:
            #test_tensor = test_img.unsqueeze(0) * 2 - 1
            #train_tensor = train_img.unsqueeze(0) * 2 - 1
            #with torch.no_grad():
            #    d = loss_fn(test_tensor, train_tensor).item()
            tensor_img1 = train_img.to(device)
            
            d = lpips_distance(loss_fn, tensor_img0, tensor_img1)
            dists.append(d)
        # Find the 3 most similar images
        dists.sort()
        top3 = dists[:3]
        mean_dist = sum(top3) / 3
        distances_per_test.append(mean_dist)
    overall_mean = sum(distances_per_test) / len(distances_per_test)
    print(f"Transformation: {t_name}, Mean LPIPS distance: {overall_mean:.4f}")

    
    new_train_dataset = CustomListDataset(new_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy =train_model(model, train_loader, test_loader, device)

    return overall_mean, accuracy
    
# Experiment contamination_with_specific_transformation_helper
# For each transformation, add the transformed test image to training set,
# then for each test image compute the LPIPS distance to the 3 most similar training samples.
# The average of these distances across test images is calculated per transformation.
def contamination_with_specific_transformation(train_dataset, test_dataset, model, batch_size):
    print("[Experiment 5] Transformation Impact Analysis")
    transformations = get_transformations(train_dataset[0][0])
    results = {}
        
    # For each transformation check its impact separately:
    for t_name, transform in transformations.items():
        print(f"\nProcessing transformation: {t_name}")
        overall_mean, accuracy = contamination_with_specific_transformation_helper(train_dataset, test_dataset, model, t_name)
        results[t_name] = (overall_mean, accuracy)
       
    print("\nOverall Transformation Impact Results:")
    for t_name, (score,accuracy) in results.items():
        print(f"{t_name}: mean={score:.4f} accuracy={accuracy:.2f}")


def contamination_with_specific_transformation_without_mean_distance_helper(train_dataset, test_dataset, model, t_name): 
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
    
    overall_mean = 0
    print(f"Transformation: {t_name}, Mean LPIPS distance: {overall_mean:.4f}")

    new_train_dataset = CustomListDataset(new_train_examples)
    train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = train_model(model, train_loader, test_loader, device)

    return overall_mean, accuracy
    

# Experiment contamination_with_specific_transformation_helper
# For each transformation, add the transformed test image to training set
# Without calculating the mean LPIPS distance
def contamination_with_specific_transformation_without_mean_distance(train_dataset, test_dataset, model):
    print("[Experiment 5] Transformation Impact Analysis")
    transformations = get_transformations(train_dataset[0][0])
    results = {}
        
    # For each transformation check its impact separately:
    for t_name, transform in transformations.items():
        print(f"\nProcessing transformation: {t_name}")
        overall_mean, accuracy = contamination_with_specific_transformation_without_mean_distance_helper(train_dataset, test_dataset, model, t_name)
        results[t_name] = (overall_mean, accuracy)
       
    print("\nOverall Transformation Impact Results:")
    for t_name, (overall_mean,accuracy) in results.items():
        print(f"{t_name}: mean={overall_mean:.4f} accuracy={accuracy:.2f}")


def main():
    if(len(sys.argv) < 3):
        print("wrong usage: python3 experiments.py <1/2/3/4/5> <alexnet/resnet50> [<thresold>/<num_neigh_to_remove>]")
        exit(-1)

    data_dir = "./cifar_images_train_test/CIFAR-10-images/"
    experiment = int(sys.argv[1])        # 1,2,3,4,5,6
    model = sys.argv[2]     # "alexnet" or "resnet50"

    if(experiment == 4):
        if(len(sys.argv) < 4):
            threshold = 0.5
        else:
            threshold = float(sys.argv[3])
    elif(experiment == 5):
        if(len(sys.argv) < 4):
            neigh_to_remove = 6
        else:
            neigh_to_remove = int(sys.argv[3])

    train_dataset, excluded_train_dataset, test_dataset = get_datasets(data_dir)
    if experiment == 1:
        clean_training(train_dataset, test_dataset, model)
    elif experiment == 10:
        clean_training_with_mean_lpips_distance(train_dataset, test_dataset, model)
    elif experiment == 2:
        exact_duplicates_contamination(train_dataset, test_dataset, model)
    elif experiment == 20:
        exct_duplicates_with_repetition_contamination(train_dataset, test_dataset, model)
    elif experiment == 3:
        near_duplicates_contamination(train_dataset, test_dataset, model)
    elif experiment == 4:
        decontamination_removing_n_duplicates(train_dataset, excluded_train_dataset, test_dataset, model, neigh_to_remove)
    elif experiment == 5:
        decontamination_with_threshold(train_dataset, excluded_train_dataset,test_dataset, model, threshold)
    elif experiment == 6:
        contamination_with_specific_transformation(train_dataset, test_dataset, model)
    elif experiment == 60:
        contamination_with_specific_transformation_without_mean_distance(train_dataset, test_dataset, model)
        
if __name__ == "__main__":
    main()
