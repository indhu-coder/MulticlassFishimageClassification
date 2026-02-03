VGG16 is a Convolutional Neural Network (CNN) architecture introduced in 2014 by Simonyan & Zisserman.

It is 16 layers deep — hence the name VGG16:
     13 convolutional layers
     3 fully connected layers at the end.
     
Famous for:

-->Very uniform architecture (all conv layers are 3×3)

-->Good feature extraction

-->Easy to use for transfer learning

<img width="1536" height="1024" alt="ChatGPT Image Jan 22, 2026, 12_38_49 AM" src="https://github.com/user-attachments/assets/526883ee-0ed0-4879-8a97-df344b9e1aee" />

Loading the model and preprocessing the image are given below:

     from sklearn.metrics import classification_report
     import torch
     import torch.nn as nn
     import torch.optim as optim
     from torch.utils.data import DataLoader
     from torchvision import datasets, models,transforms
     import numpy as np
     from PIL import Image
     from torchvision.models import  vgg16, VGG16_Weights
     import os
     from sklearn.metrics import confusion_matrix
     import seaborn as sns
     import matplotlib.pyplot as plt
     
     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
     
     TRAIN_DIR = "D:/Multiclass Fish Image classification/images.cv/data/train"
     VAL_DIR   = "D:/Multiclass Fish Image classification/images.cv/data/val"
     TEST_DIR  = "D:/Multiclass Fish Image classification/images.cv/data/test"
     
     BATCH_SIZE = 64
     EPOCHS = 5
     LR = 1e-3
     T = 3.5  
     
     weights = models.VGG16_Weights.DEFAULT
     model = vgg16(weights=weights)
     train_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )
     ])
     
     val_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )
     ])
     
     test_transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )
     ])
     
     train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
     val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_transform)
     test_ds  = datasets.ImageFolder(TEST_DIR, transform=test_transform)
     
     class_names = train_ds.classes
     NUM_CLASSES = len(class_names)
     
     # # print(train_ds.class_to_idx)
     # # print(test_ds.class_to_idx)
     
     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=False)
     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
     test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

     # #  FREEZE ALL layers
     for param in model.features.parameters():
         param.requires_grad = False
     # #  Replace classifier layers
     model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
     # #  Training setup
     model = model.to(DEVICE)
     criterion = nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

The architecture of VGG 16 model  after freezing the featues and replacing the classifier as per our fish dataset is shown below:

 (features): Sequential(
(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
(1): ReLU(inplace=True)
    
(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(3): ReLU(inplace=True)

(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(6): ReLU(inplace=True)

(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(8): ReLU(inplace=True)

(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(11): ReLU(inplace=True)

(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(13): ReLU(inplace=True)

(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(15): ReLU(inplace=True)

(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(18): ReLU(inplace=True)

(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(20): ReLU(inplace=True)

(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(22): ReLU(inplace=True)

(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(25): ReLU(inplace=True)

(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(27): ReLU(inplace=True)

(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(29): ReLU(inplace=True)

(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  
  (classifier): Sequential(

(0): Linear(in_features=25088, out_features=4096, bias=True)

(1): ReLU(inplace=True)

(2): Dropout(p=0.5, inplace=False)

(3): Linear(in_features=4096, out_features=4096, bias=True)

(4): ReLU(inplace=True)

(5): Dropout(p=0.5, inplace=False)

(6): Linear(in_features=4096, out_features=11, bias=True)
  )
)


Traing the model with train dataloader and validation dataloader is given below:

     for epoch in range(EPOCHS):
     #     model.train()
     
     #     train_loss = 0
     
     #     for imgs, labels in train_loader:
     #         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
     #         optimizer.zero_grad()
     #         outputs = model(imgs)
     #         loss = criterion(outputs, labels)
     #         loss.backward()
     #         optimizer.step()
     #         train_loss += loss.item()
             
     
     
     # # # #  ---------- Validation ----------
     # model.eval()
     # val_loss, correct, total = 0, 0, 0
     # # energy_vals = []
     # energies = []  
     # with torch.no_grad():
     #         for imgs, labels in val_loader:
     #             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
     #             outputs = model(imgs)
     #             loss = criterion(outputs, labels)
     #             val_loss += loss.item()
     #             preds = outputs.argmax(dim=1)
     #             correct += (preds == labels).sum().item()
     #             total += labels.size(0)
     #             val_acc = 100 * correct / total
     #         print(f"Epoch {epoch+1}/{EPOCHS} | "
     #           f"Train Loss: {train_loss/len(train_loader):.4f} | "
     #           f"Val Loss: {val_loss/len(val_loader):.4f} | "
     #           f"Val Acc: {val_acc:.2f}%")


Since VGG 16 is too slow and for 2 epoch accuracy was 11% epoch was increased to 5 and accuracy result are as follows;

     Epoch 5/5 | Train Loss: 0.5522 | Val Loss: 0.3698 | Val Acc: 98.17%

Why VGG16 feels painfully slow?

-->138 million parameters

-->Huge fully connected layers (4096 units × 2)

--->No depthwise separable convs

--->Designed before efficiency was a priority

After obtaining maximum accuracy the model has been saved in pickle file

                # Save the best model
               # torch.save({
               #     "model_state": model.state_dict(),
               #     "class_names": class_names,
               #    }, "models/vgg16_best.pkl")
               
               checkpoint = torch.load(
                   "models/vgg16_best.pkl",
                   map_location=DEVICE,weights_only=False
                   )
               
               model.load_state_dict(checkpoint["model_state"])
               # model.eval()
               # print(model)
               class_names = checkpoint["class_names"]

Checking the training accuracy and Test accuracy with the saved model is for better understanding of the model.

     To check train accuracy
     # # model.eval()
     # correct, total = 0, 0
     
     # with torch.no_grad():
     #     for x, y in train_loader:
     #         x, y = x.to(DEVICE), y.to(DEVICE)
     #         preds = model(x).argmax(dim=1)
     #         correct += (preds == y).sum().item()
     #         total += y.size(0)
     
     # print("Train accuracy:", 100 * correct / total)
     
     # Test the model
     # correct, total = 0, 0
     # # ENERGY_T = -2.2595994
     # y_true = []
     # y_pred = []
     # # UNKNOWN_IDX = len(class_names)
     # all_class_names = class_names
     # with torch.no_grad():
     #     for imgs, labels in test_loader:
     #         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
     #         logits = model(imgs)
     #         # energy = energy_score(logits, T=T)
     
     #         preds = logits.argmax(dim=1)
     #         # preds[energy > ENERGY_T] = UNKNOWN_IDX
     #         correct += (preds == labels).sum().item()
     #         total += labels.size(0)
     #         y_true.extend(labels.cpu().numpy())
     #         y_pred.extend(preds.cpu().numpy())
     
     # print("Accuracy:", 100 * correct / total)

Accuracies are

Train accuracy: 99.21285140562249

Test Accuracy: 98.39974898023219

Since the accuracies are higher classification report was taken alongwith confusion matrix for each classes.


     --- Classification Report (Test Data) ---
                                       precision    recall  f1-score   support
     
                          animal fish       1.00      0.97      0.98       520
                     animal fish bass       0.50      0.92      0.65        13
        fish sea_food black_sea_sprat       1.00      1.00      1.00       298
        fish sea_food gilt_head_bream       0.93      1.00      0.96       305
        fish sea_food hourse_mackerel       1.00      0.99      0.99       286
             fish sea_food red_mullet       1.00      1.00      1.00       291
          fish sea_food red_sea_bream       1.00      0.93      0.97       273
               fish sea_food sea_bass       0.99      0.98      0.98       327
                 fish sea_food shrimp       0.98      1.00      0.99       289
     fish sea_food striped_red_mullet       1.00      0.98      0.99       293
                  fish sea_food trout       0.99      1.00      1.00       292
     
                             accuracy                           0.98      3187
                            macro avg       0.94      0.98      0.96      3187
                         weighted avg       0.99      0.98      0.98      3187

<img width="640" height="480" alt="confusion matric vgg 16" src="https://github.com/user-attachments/assets/13515a2d-998f-4139-b22b-5344d5d6b75a" />

For predicting the single image Inference code is given below:

          ENERGY_T = -1.5
          T = 2.5
          MARGIN_T = 1.0
          
          def predict_image(image_path):
              model.eval()
          
              image = Image.open(image_path).convert("RGB")
              image = test_transform(image).unsqueeze(0).to(DEVICE)
          
              with torch.no_grad():
                  logits = model(image)
                  logits = logits / T
          
                  probs = torch.softmax(logits, dim=1)
                  confidence = probs.max(dim=1).values.item()
          
                  energy = -torch.logsumexp(logits, dim=1).item()
          
                  pred = logits.argmax(dim=1).item()
          
                  # Logit margin
                  top2 = torch.topk(logits, 2, dim=1).values
                  margin = (top2[:, 0] - top2[:, 1]).item()
          
                  # UNKNOWN decision (FIXED)
                  if energy > ENERGY_T or margin < MARGIN_T:
                      class_name = "unknown"
                  else:
                      class_name = class_names[pred]
          
              return class_name, confidence, energy, margin
          
          
          
          image_path = "D:/Multiclass Fish Image classification/images.cv/data/9.jpg"
          
          class_name, confidence, energy, margin = predict_image(image_path)
          print(f"Predicted Class : {class_name}")
          print(f"Confidence      : {confidence * 100:.2f}%")
          print(f"Energy          : {energy:.4f}")
          print(f"Margin          : {margin:.4f}")
          print(f"Threshold       : {ENERGY_T}")

And the final result is

Predicted Class : fish sea_food trout
Confidence      : 100.00%
Energy          : -21.8808
Margin          : 21.6927
Threshold       : -1.5

Predicted Class : fish sea_food shrimp
Confidence      : 100.00%
Energy          : -68.2732
Margin          : 73.8294
Threshold       : -1.5


Limitations:

1.After experimenting with three pre-trained models the differnce between the models in terms of speed is given below:

          Model	          Params	Speed
          InceptionV3	     ~23M	     Fast
          ResNet50	          ~25M     	Fast
          VGG16	          138M	     Slow

2. Energy-based and margin-based OOD methods failed due to severe overconfidence caused by dataset bias and representation collapse. Incorporating outlier exposure or an explicit unknown class is necessary.

For predicting the known classes VGG 16 can be used but accleration speed has to be kept in mind.

Future Improvements:

In future if unknown images are incorporated with the dataset and trained with this model with energy based method it can predict the image with higher accuracy rate.







     
