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

Loss and accuracy plots have been saved in the Plots folder.

### Hyperparameters
The following hyperparameters were selected based on standard practices for fine-tuning pretrained convolutional neural networks and empirical evaluation on the validation set.

| Hyperparameter           | Value                   | Description                                                             |
|--------------------------|--------------------------|-------------------------------------------------------------------------|
| Optimizer                | SGD                      | Stochastic Gradient Descent optimizer used for training                 |
| Learning Rate            | 0.01                     | Initial learning rate                                                   |
| Momentum                 | 0.9                      | Helps accelerate gradient updates and dampens oscillations              |
| LR Scheduler             | ReduceLROnPlateau        | Reduces the learning rate when a metric has stopped improving           |
| Loss Function            | CrossEntropyLoss         | Suitable for multi-class classification tasks                           |
| Batch Size               | 32                       | Number of samples processed before the model is updated                 |


### Results (on Validation Set)
| Model         | Parameters |  Image Input     | Validation Accuracy |
|---------------|------------|------------------|---------------------|
| ResNet-18     |    11M     | Full Image       | 66.3%               |
| ResNet-18     |    11M     | Cropped (BBox)   | 68.5%               |
| ResNet-50     |    27M     |  Full Image      | **79.6%**           |

### Results (on test data)
- Overall_accuracy (%): 76.13048
- max_accuracy_class: 133
- max_accuracy (%): 100.0
- min_accuracy_class: 101
- min_accuracy (%)': 16.66667
  
Predictions made by ResNet50 (which was submitted) are stored in pred.csv

### Conclusion
The highest validation accuracy of 79.6% was achieved using ResNet-50. The test set performance (76.13%) closely aligned with validation accuracy, suggesting good generalization. Unlike standard image classification tasks where inter-class differences are often large and easily separable (between a bird and a dog), fine-grained visual classification requires distinguishing between categories that exhibit high inter-class similarity and low intra-class variability. In such scenarios, the model must go beyond extracting low-level features (edges, color blobs, basic shapes) and instead learn to identify subtle, localized, and discriminative patterns. 
Moreover, each class in this dataset contains only ~30 images, which is insufficient to train a deep network from scratch. To address this, transfer learning was employed using pretrained models (ResNet-18 and ResNet-50) as feature extractors, using the representations learned from the large ImageNet dataset. While this approach significantly improves convergence and generalization, the limited data and use of relatively standard architectures still constrain the model's ability to fully capture the distinctions required for fine-grained classification, resulting in only moderate performance.
