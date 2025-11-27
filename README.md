Image Classification is a core computer vision task that involves assigning a predefined label to an image based on its visual content. 
This project focuses on classifying fish images into multiple categories using deep learning models. 
The task involves training a CNN from scratch and leveraging transfer learning with pre-trained models to enhance performance. 
The project also includes saving models for later use and deploying a Streamlit application to predict fish categories from user-uploaded images.

Data Collection and Preprocessing:
The dataset consists of images of fish, categorized into folders by species. 
The dataset is loaded using TensorFlow's ImageDataGenerator for efficient processing.
Gathering images and applying techniques like resizing, normalization, and data augmentation.

1. Define the transformations you want to apply
   
              # These transforms are applied on-the-fly to every image
              data_transforms = transforms.Compose([
                  transforms.Resize((128, 128)),  # Resize all images to 128x128
                  transforms.RandomRotation(degrees=(-10, 10)),  #Rotating the images    
                  transforms.RandomHorizontalFlip(p=0.5), # flippin the images   
                  transforms.Grayscale(num_output_channels=1), # Convert to grayscale
                  transforms.ToTensor(),                  # Convert PIL Image to Tensor (float32, 0-1 range)
                  transforms.Normalize((0.5,), (0.5,)),   #  added normalization here: )
                 
              ])
   
3. Instantiate the ImageFolder dataset
   
              # Just provide the path to the 'root' directory (e.g., './data/train')
              train_dataset = datasets.ImageFolder(
                  root="D:/Multiclass Fish Image classification/images.cv/data/train",      
                  transform=data_transforms
              )
   
3. Create the DataLoader
   
              # The DataLoader wraps the dataset and handles batching and shuffling
              train_dataloader = DataLoader(
                  train_dataset,
                  batch_size=64,
                  shuffle=True,
                 
              )
4. Accessing information
   
            # print(f"Found {len(train_dataset)} images in the dataset.")
            # print(f"Classes found: {train_dataset.classes}")
            # print(f"Class to index mapping: {train_dataset.class_to_idx}")

   The result :

   Found 6225 images in the dataset.
   
   Classes found: ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']
   
   Class to index mapping: {'animal fish': 0, 'animal fish bass': 1, 'fish sea_food black_sea_sprat': 2, 'fish sea_food gilt_head_bream': 3, 'fish sea_food hourse_mackerel': 4, 'fish sea_food red_mullet': 5, 'fish sea_food red_sea_bream': 6, 'fish sea_food sea_bass': 7, 'fish sea_food shrimp': 8, 'fish sea_food striped_red_mullet': 9, 'fish sea_food trout': 10}
   
5.Model Building

The field of machine learning has taken a dramatic twist in recent times, with the rise of the Artificial Neural Network (ANN). Thesebiologically inspired computational models are able to far exceed the performance of previous forms of artificial intelligence in common machine learning tasks. One of the most impressive forms of ANN architecture is that of the Convolutional Neural Network (CNN). CNNs are primarily used to solve difficult image-driven pattern recognition tasks and with their precise yet simple architecture, offers a simplified method of getting started with ANNs.


CNNs are comprised of three types of layers. These are convolutional layers, pooling layers and fully-connected layers. When these layers are stacked, a CNN architecture has been formed.

![cnn layers](https://github.com/user-attachments/assets/e5b49fe9-6f26-43eb-b408-645ae6dcab28)

for further in-depth knowledge kindly refer the paper attched below:
https://arxiv.org/pdf/1511.08458

Code for the same architecture given below:

      class CNN (nn.Module):
      #     def __init__(self):
      #         super().__init__()
      #         self.conv1 = nn.Conv2d(in_channels= 1,out_channels=32,kernel_size=3,padding=1,stride=1)
      #         self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
      #         self.conv2 = nn.Conv2d(in_channels= 32,out_channels=64,kernel_size=3,padding=1,stride=1)
              
      #         self.relu = nn.ReLU()
      #         self.fc1 = nn.Linear(in_features = 64*32*32,out_features = 128) 
      #         self.fc2 = nn.Linear(in_features = 128,out_features = 11)
      
      #     def forward(self,x):
      #         conv1 = self.pool(self.relu(self.conv1(x))) 
      #         conv2 = self.pool(self.relu(self.conv2(conv1)))
              
      #         flatten = conv2.view(-1,64*32*32)
      #         fc1 = self.relu(self.fc1(flatten))
      #         output = self.fc2(fc1)
      #         return output

6.The Training phase
  
      model = CNN()
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(),lr=0.001)
      train_losses_history = []
      val_losses_history = []
      val_accuracy_history = []
      
# Initialize a variable to keep track of the best validation loss found so far
# Start with a high number so the first epoch is always better
best_val_loss = float('inf') 
checkpoint_path = 'custom_cnn_checkpoint.pth' # File path to save the best model weights

# # # Training the model
      num_epochs = 20

      for epoch in range(num_epochs):
         # --- 1. TRAINING PHASE ---
          model.train()
          current_epoch_train_losses = []
             for images,labels in train_dataloader:
                  # images = images.float()
                  # # print(f"Batch of images shape: {images.shape}")
                  
                  optimizer.zero_grad() #` Zero the gradients`
                  predicted_outputs = model(pixel_values=images,labels = labels)  # Forward pass
                
                  # print(predicted_outputs)
                  loss = criterion(predicted_outputs,labels)
                  # Backpropagation and optimization
                  
                  
                  loss.backward() # Computation of the Gradient
                  optimizer.step() # Step by step updates the weights according to the gradient
      
                  current_epoch_train_losses.append(loss.item())
        
    # Calculate average training loss for this epoch
    avg_train_loss = sum(current_epoch_train_losses) / len(current_epoch_train_losses)
    train_losses_history.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
      


