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
      
Initialize a variable to keep track of the best validation loss found so far
Start with a high number so the first epoch is always better
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
      
7.Validation Phase 

Function of validation phase: 

      def evaluate_validation_set(model, val_dataloader, criterion):
          # Ensure the model is in evaluation mode
          model.eval()
    
          total_loss = 0.0
          correct_predictions = 0
          total_samples = 0
          
          # We use torch.no_grad() because we don't need to calculate gradients during validation
          with torch.no_grad():
              for images, actual_labels in val_dataloader:
                  # 1. Forward pass
                  outputs = model(pixel_values=images, labels=actual_labels)
                  
                  # Calculate batch loss (optional, but good practice)
                  loss = criterion(outputs, actual_labels)
                  
                  total_loss += loss.item() * images.size(0) # Accumulate weighted loss
                  
                  # 2. Get predictions (indices of the max probability)
                  _, predicted_labels = torch.max(outputs.data, 1)
                  
                  # 3. Track total samples
                  batch_size = actual_labels.size(0)
                  total_samples += batch_size
                  
                  # 4. Track correct predictions
                  # Compare predicted indices with actual labels
                  correct_predictions += (predicted_labels == actual_labels).sum().item()
                  
                  # --- Optional: Print batch-specific metrics ---
                  # print(f'Batch Size: {batch_size}, Correct: {((predicted_labels == actual_labels).sum().item() / batch_size) * 100:.2f}% Accuracy')
                  # print('************************************')

    # Calculate average loss and total accuracy for the whole epoch/dataset
    avg_loss = total_loss / total_samples
    overall_accuracy = (correct_predictions / total_samples) * 100.0
    
    return avg_loss, overall_accuracy
    
    model.eval()
    current_epoch_val_losses = []
         # Use the evaluation function we discussed earlier
    avg_val_loss, overall_accuracy = evaluate_validation_set(model, val_dataloader, criterion)
    
    val_losses_history.append(avg_val_loss)
    val_accuracy_history.append(overall_accuracy)
    
--- Checkpointing Logic: Save the model if this epoch is the best so far ---

    if avg_val_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model checkpoint...")
                best_val_loss = avg_val_loss
        
        # Save the model's parameters (state_dict) to a file
                torch.save(model.state_dict(), checkpoint_path)

 --- LOGGING AND SAVING ---
 
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Validation Loss: {avg_val_loss:.4f}, '
          f'Validation Accuracy: {overall_accuracy:.2f}%')   

      val_loss, val_accuracy = evaluate_validation_set(model, val_dataloader, criterion)
      print(f'Avg Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

Data provided from the training log for CNN model represented in picture

train_losses = [
    1.5994, 0.9242, 0.6726, 0.5453, 0.4487, 0.3624, 0.2948, 0.2354, 0.2122, 0.1833,
    0.1550, 0.1278, 0.1239, 0.0936, 0.0807, 0.0889, 0.0795, 0.0625, 0.0409, 0.0668
]

validation_losses = [
    1.3709, 0.8149, 0.6736, 0.5606, 0.4777, 0.4917, 0.3777, 0.3288, 0.2901, 0.2758,
    0.2483, 0.2764, 0.2282, 0.1862, 0.2368, 0.2470, 0.2662, 0.1939, 0.1987, 0.2097
]

      # epochs = range(1, len(train_losses) + 1)
      
      # # Highlight the optimal stopping point (around Epoch 14)
      # optimal_epoch = 14
      # optimal_val_loss = validation_losses[optimal_epoch - 1]
      
      # Plotting the data
      # plt.figure(figsize=(10, 6))
      # plt.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='-', linewidth=2)
      # plt.plot(epochs, validation_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)
      
      # # Mark the optimal stopping point
      # plt.axvline(x=optimal_epoch, color='green', linestyle=':', linewidth=2, label=f'Optimal Stopping Point (Epoch {optimal_epoch})')
      # plt.scatter(optimal_epoch, optimal_val_loss, color='green', s=100, zorder=5)
      
      # Adding titles and labels
      # plt.title('Training and Validation Loss Over Epochs (Learning Curve) for CNN Model')
      # plt.xlabel('Epoch')
      # plt.ylabel('Loss Value')
      # plt.legend()
      # plt.grid(True, which='both', linestyle=':', linewidth=0.5)
      
      # # Set x-axis ticks to show every epoch clearly
      # plt.xticks(epochs)
      
      # # Display the plot
      # plt.tight_layout()
      # plt.show()
      
<img width="1280" height="612" alt="Learning curve final" src="https://github.com/user-attachments/assets/0fb66438-601a-4964-862c-52f4e340381b" />


Load the weights from the OPTIMAL Epoch (Epoch 14 weights)

      # try:
      #     model.load_state_dict(torch.load(CHECKPOINT_PATH))
      #     print(f"Successfully loaded optimal model weights from {CHECKPOINT_PATH}")
      # except FileNotFoundError:
      #     print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}. Cannot run test evaluation.")
      #     exit()
8.Testing phase

       test_dataset = datasets.ImageFolder(root = "D:/Multiclass Fish Image classification/images.cv/data/test",      # Replace with your actual root path
        #     transform=data_transforms
        # test_dataloader = DataLoader(
        #     test_dataset,batch_size=64,shuffle=True)

        # # def evaluate_test_set(model, test_dataloader, criterion):
        # #     # Ensure the model is in evaluation
        # #     model.eval()
        # #     total_loss = 0.0
        # #     correct_predictions = 0
        # #     total_samples = 0
        # #         # Lists to store ALL predictions and actuals for Confusion Matrix/Metrics
        # #     all_predicted_labels = []
        # #     all_actual_labels = []

        # #     with torch.no_grad():
        # #         for images, actual_labels in test_dataloader:
        # #             outputs = model(pixel_values=images, labels=actual_labels)
        # #             loss = criterion(outputs, actual_labels)
        # #             total_loss += loss.item() * images.size(0)
        # #             _, predicted_labels = torch.max(outputs.data, 1)
        # #             batch_size = actual_labels.size(0)
        # #             total_samples += batch_size
        # #             correct_predictions += (predicted_labels == actual_labels).sum().item()

                    
        # #             # --- Capture all labels for metrics ---
        # #             all_predicted_labels.extend(predicted_labels.tolist())
        # #             all_actual_labels.extend(actual_labels.tolist())

        # #             avg_loss = total_loss / total_samples
        # #             overall_accuracy = (correct_predictions / total_samples) * 100.0
        # #         return avg_loss, overall_accuracy, all_predicted_labels, all_actual_labels
            

        # # # Run the evaluation
        # # test_loss, test_accuracy, predictions, actuals = evaluate_test_set(model, test_dataloader, criterion)

        # # print(f'Final Test Loss: {test_loss:.4f}')
        # # print(f'Final Test Accuracy: {test_accuracy:.2f}%')

The output is 
Successfully loaded optimal model weights from custom_cnn_checkpoint.pth
Final Test Loss: 0.1705
Final Test Accuracy: 94.79%

9.Evaluation Metrics

      # Generate the Classification Report (Precision, Recall, F1-Score)
      print("\n--- Classification Report (Test Data) ---")
      
      target_names = [train_dataset.classes[i] for i in range(len(train_dataset.classes))]
      print(classification_report(actuals, predictions, target_names=target_names))
      
      # Generate and Plot the Confusion Matrix
      cm = confusion_matrix(actuals, predictions)
      plt.figure(figsize=(10, 8))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
      plt.xlabel('Predicted Label')
      plt.ylabel('True Label')
      plt.title('Confusion Matrix for Test Set')
      plt.show()

and the output is

      --- Classification Report (Test Data) ---
                                        precision    recall  f1-score   support
      
                           animal fish       0.97      0.95      0.96       520
                      animal fish bass       0.00      0.00      0.00        13
         fish sea_food black_sea_sprat       0.91      0.99      0.95       298
         fish sea_food gilt_head_bream       0.95      0.94      0.95       305
         fish sea_food hourse_mackerel       0.92      0.97      0.94       286
              fish sea_food red_mullet       0.96      0.91      0.93       291
           fish sea_food red_sea_bream       0.96      0.95      0.95       273
                fish sea_food sea_bass       0.93      0.92      0.93       327
                  fish sea_food shrimp       0.99      0.98      0.98       289
      fish sea_food striped_red_mullet       0.93      0.92      0.93       293
                   fish sea_food trout       0.98      0.99      0.99       292
      
                              accuracy                           0.95      3187
                             macro avg       0.86      0.87      0.86      3187
                          weighted avg       0.95      0.95      0.95      3187

<img width="1280" height="612" alt="confusion matrix final" src="https://github.com/user-attachments/assets/a00d7bb0-958e-4c4b-9703-624468083faf" />

Now its time to test a single image prediction using customised CNN architecture.

      def predict_single_image(model, image_path, transform, class_names, threshold=0.65,T=10.0):
         #     from PIL import Image
         #     # Load and preprocess the image
         #     image = Image.open(image_path) # Ensure it's in RGB format
             
         #     transform = transforms.Compose([
         #         transforms.Grayscale(num_output_channels=1),
         #          transforms.Resize((128,128)),
         #          transforms.ToTensor(),
         #             transforms.Normalize((0.5,),(0.5,)),
         #              transforms.RandomRotation(degrees = (-20,+20))
         #     ])
         #     image = transform(image).unsqueeze(0)  # Add batch dimension
         #     # Set model to evaluation mode
         #     model.eval()
             
         #     with torch.no_grad():
         #          # Forward pass
         #         logits = model(image)
         
                
         
         #        # Apply temperature scaling
         #         scaled_logits = logits / T
         
         #         # Compute softmax with temperature
         #         probs = F.softmax(scaled_logits, dim=1)
         
         #         # Get max probability and predicted class index
         #         max_prob, pred_idx = torch.max(probs, dim=1)
         #         max_prob = max_prob.item()
         #         pred_idx = pred_idx.item()
         
           
         
         #         # Check threshold
         #         if max_prob < threshold:
         #             return class_names[pred_idx], max_prob
         #         else:
         #             return "Unknown",max_prob
                 
         
         
         # # Example usage:
         # image_path = 'D:/Multiclass Fish Image classification/images.cv/data/4.jpg'
         # CLASS_NAMES = train_dataset.classes  # Assuming you have this from your dataset
         # predicted_label, confidence_score = predict_single_image(model, image_path, transforms,CLASS_NAMES,threshold = 0.65,T=10.0)
         
         # print(f"The model predicted: {predicted_label} with {confidence_score:.2f}% confidence.")
         
         # plt.imshow(imread(image_path))
         # plt.axis('off')
         # plt.title(f'Predicted Class: {predicted_label} ({confidence_score:.2f}%)')
         # plt.show()



