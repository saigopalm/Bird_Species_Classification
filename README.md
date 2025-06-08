# Bird Species Classification

### Understanding Data
This project is focused on classifying images of birds into one of 200 species in the given dataset. Since this is a multi-class classification problem, a critical first step is to examine whether the dataset suffers from class imbalance. A significant imbalance could bias the model toward over-represented classes. The summary statistics shown below indicate that data is well balanced across classes.
```bash
Summary of class distribution:
Min samples in a class:   29  
Max samples in a class:   30  
Mean samples per class:   29.97  
Std deviation:            0.17101529509309654
```
Despite being balanced, the dataset presents a high level of complexity due to its limited number of samples per class (~30) combined with a large number of classes (200 bird species). This low-data, high-class-count scenario makes the classification task challenging, as the model must learn fine-grained visual distinctions with very few examples per category.

### Creating Validation Set
Since the ground truth labels for the test dataset are not provided, the original training data was split (90% training, 10% validation) into training and validation sets. A stratified split was used to ensure that each class was proportionally represented in both sets.

### Data Loading and Preprocessing
A custom Dataset class was implemented to handle image loading and preprocessing. As part of the preprocessing pipeline
- Images were resized to 224×224 pixels.
- A random horizontal flip was applied with a probability of 0.5 to introduce basic data augmentation.
- Images were then converted to PyTorch tensors and normalized. 

### Model Selection and Training
To begin with two standard architectures, ResNet-18 and ResNet-50 pretrained on ImageNet dataset were adopted. The final fully connected layer of each network was modified to output logits over 200 classes. An important aspect of the given dataset is that each image contains exactly one bird, and bounding box coordinates for each instance are provided. To exploit this,
- Each image was tightly cropped using the bounding box coordinates.
- This eliminates irrelavant background and enhancing the model's focus on object of interest - bird.
- After cropping, the image was resized and normalized using standard ImageNet statistics.

### Results (on Validation Set)
| Model         | Parameters |  Image Input     | Validation Accuracy |
|---------------|------------|------------------|---------------------|
| ResNet-18     |    11M     | Full Image       | 66.3%               |
| ResNet-18     |    11M     | Cropped (BBox)   | 68.5%               |
| ResNet-50     |    27M     |  Full Image      | **79.6%**           |
