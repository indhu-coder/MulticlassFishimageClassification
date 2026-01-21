Multiclass Fish Image Classification using RESNET-50 Architecture(Pre-Trained model)


Pre-Trained model from Hugging face:
ResNet (Residual Network) is a convolutional neural network that democratized the concepts of residual learning and skip connections. This enables to train much deeper models.
This is ResNet v1.5, which differs from the original model: in the bottleneck blocks which require downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution. 


Pre-processing the images for the pre-trained model:
        
            def preprocess_image(image):
              #Setting return_tensors="pt" ensures it returns PyTorch tensors
            processed_image = processor(image, return_tensors="pt")['pixel_values'][0]
    	        #The [0] is to remove the batch dimension the processor adds by default
            return processed_image

Calling the model from Hugging Face:

            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50",use_fast=True)
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

Training the model:

       # set device to GPU if available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # print(f"Using device: {device}")
        model = model.to(device)  # Move model to CPU (or 'cuda' for GPU if available)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     train_losses_history = []
    #     val_losses_history = []
    #     val_accuracy_history = []
     num_epochs = 1
    #     for epoch in range(num_epochs):
    #         # --- 1. TRAINING PHASE ---
    #             model.train()
    #             running_loss = 0.0
    #             current_epoch_train_losses = []
    #             for images,labels in train_dataloader:
    #                 # images = images.float()
    #                 # print(f"Batch of images shape: {images.shape}")
    #                 images,labels = images.to(device),labels.to(device)
    #                 optimizer.zero_grad() #` Zero the gradients`
    #                 predicted_outputs = model(pixel_values=images,labels = labels)  # Forward pass
                
    #                 # print(predicted_outputs)
    #                 loss=predicted_outputs.loss
               
    #                 # Backpropagation and optimization
    #                 loss.backward()
    #                 optimizer.step()

    #                 running_loss += loss.item()
                   
    #     train_loss = running_loss / len(train_dataloader)
    #     current_epoch_train_losses.append(loss.item())
            
    #     # #     # Calculate average training loss for this epoch
    #     avg_train_loss = sum(current_epoch_train_losses) / len(current_epoch_train_losses)
    #     train_losses_history.append(avg_train_loss)
    # #     # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

The results are:


                Epoch [1/1], Loss: 0.0067
                Using device: cpu
               

Next comes the validation phase.
2. VALIDATION PHASE 

best_val_loss = float('inf') 
checkpoint_path_RESNET = 'resnet_fish_checkpoint1.pth'   
def evaluate_validation_set(model, val_dataloader):
    #         # Ensure the model is in evaluation mode
    #         model.eval()
            
    #         total_loss = 0.0
    #         correct_predictions = 0
    #         total_samples = 0
    #         all_energy = []
    #         # We use torch.no_grad() because we don't need to calculate gradients during validation
    #         with torch.no_grad():
    #             for images, actual_labels in val_dataloader:
    #                 # 1. Forward pass
    #                 outputs = model(pixel_values=images, labels=actual_labels)
    #                 logits = outputs.logits
    #                 # Calculate batch loss (optional, but good practice)
    #                 loss = outputs.loss
                    
    #                 total_loss += loss.item() * images.size(0) # Accumulate weighted loss
                    
    #                 # 2. Get predictions (indices of the max probability)
    #                 _, predicted_labels = torch.max(outputs.logits, 1)
                    
    #                 # 3. Track total samples
    #                 batch_size = actual_labels.size(0)
    #                 total_samples += batch_size
                    
    #                 # 4. Track correct predictions
    #                 # Compare predicted indices with actual labels
    #                 correct_predictions += (predicted_labels == actual_labels).sum().item()
                    
    #                 # --- Optional: Print batch-specific metrics ---
    #                 # print(f'Batch Size: {batch_size}, Correct: {((predicted_labels == actual_labels).sum().item() / batch_size) * 100:.2f}% Accuracy')
    #                 # print('************************************')
    #                 e = -torch.logsumexp(logits, dim=1)
    #                 all_energy.append(e)

    #         all_energy = torch.cat(all_energy)
    #         all_energy_np = all_energy.cpu().numpy()
    #         all_energy_np = all_energy_np[np.isfinite(all_energy_np)]  # remove NaN/Inf

    #         ENERGY_T =np.percentile(all_energy_np, 95)
    #         print(f'Calculated ENERGY_T: {ENERGY_T:.4f}')
    #         # Calculate average loss and total accuracy for the whole epoch/dataset
    #         avg_loss = total_loss / total_samples
    #         overall_accuracy = (correct_predictions / total_samples) * 100.0
            
    #         return avg_loss, overall_accuracy

        
    #     current_epoch_val_losses = []
    #             # Use the evaluation function we discussed earlier
    #     avg_val_loss, overall_accuracy = evaluate_validation_set(model, val_dataloader)
        
    #     val_losses_history.append(avg_val_loss)
    #     val_accuracy_history.append(overall_accuracy)

    # # --- Checkpointing Logic: Save the model if this epoch is the best so far ---
    #     if avg_val_loss < best_val_loss:
    #                 print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model checkpoint...")
    #                 best_val_loss = avg_val_loss
            
    #         # Save the model's parameters (state_dict) to a file
    #                 # torch.save(model.state_dict(), checkpoint_path_RESNET)

    # --- 3. LOGGING AND SAVING ---
    #     print(f'Epoch [{epoch+1}/{num_epochs}], '
    #             f'Train Loss: {avg_train_loss:.4f}, '
    #             f'Validation Loss: {avg_val_loss:.4f}, '
    #             f'Validation Accuracy: {overall_accuracy:.2f}%')
    #    val_loss, val_accuracy = evaluate_validation_set(model, val_dataloader)
    #     print(f'Avg Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')


The resluts are as follows:

Validation loss decreased (inf --> 0.1516). Saving model checkpoint...
Epoch [1/1], Train Loss: 0.0049, Validation Loss: 0.1089, Validation Accuracy: 98.35%
Calculated ENERGY_T: -3.8060
Avg Validation Loss: 0.1089, Validation Accuracy: 98.35%
Final Test Loss: 0.2332
Final Test Accuracy: 97.18%

Generate the Classification Report (Precision, Recall, F1-Score)
        # print("\n--- Classification Report (Test Data) ---")
        # class_names = train_dataset.classes + ["UNKNOWN"] + ["UNRECOGNISED"]
        # labels = list(range(len(class_names)))
      

        # target_names = [class_names[i] for i in range(len(class_names))]
        # print(classification_report(actuals, predictions, target_names=target_names,zero_division=0))

The result is

                                         precision    recall  f1-score   support

                             animal fish       0.99      0.96      0.98       520
                        animal fish bass       0.57      0.92      0.71        13
           fish sea_food black_sea_sprat       1.00      0.89      0.94       298
           fish sea_food gilt_head_bream       0.92      1.00      0.96       305
           fish sea_food hourse_mackerel       0.95      0.99      0.97       286
                fish sea_food red_mullet       1.00      0.99      0.99       291
             fish sea_food red_sea_bream       1.00      0.92      0.96       273
                  fish sea_food sea_bass       0.98      0.98      0.98       327
                    fish sea_food shrimp       1.00      1.00      1.00       289
        fish sea_food striped_red_mullet       0.96      0.99      0.97       293
                     fish sea_food trout       0.99      1.00      0.99       292
                                 unknown       0.00      0.00      0.00         0
                            unrecognized       0.00      0.00      0.00         0
        
                                accuracy                           0.97      3187
                               macro avg       0.80      0.82      0.80      3187
                            weighted avg       0.98      0.97      0.97      3187

Now for single image inference the code is as follows:

# def predict_single_image(model, image_path, class_names,c_score,T,ENERGY_T):
            from PIL import Image
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            # Set model to evaluation mode
            model.eval()
            
            with torch.no_grad():
                 # Forward pass
                outputs = model(pixel_values)
                logits = outputs.logits
                
                logits = logits / T  # Temperature scaling
                # Compute energy score
                energy = -torch.logsumexp(logits, dim=1).item()
                # Get predicted class and confidence
                probs = torch.softmax(logits, dim=1)
                conf_score, pred_idx = probs.max(dim=1)
                results = []
                for i in range(len(pixel_values)):
                  
                    c = conf_score[i].item()
                    idx = pred_idx[i].item()
              
                    if energy > ENERGY_T or c < c_score:
                        prediction = "UNKNOWN"
                    else:
                        prediction = class_names[idx]
                    
                    results.append({
                        'prediction': prediction,
                        'confidence': c,
                        'energy': energy
                    })
                    
            
            return results

        # Example usage:
        image_path = 'D:/Multiclass Fish Image classification/images.cv/data/11.jpg'
        class_names = train_dataset.classes  # Assuming you have this from your dataset
        results = predict_single_image(model, image_path,class_names,c_score=0.55,T=2.75,ENERGY_T = -3.8060)
        for res in results:
            print(f"Prediction: {res['prediction']}, Confidence: {res['confidence']:.4f}, Energy: {res['energy']:.4f}")



        

