MobileNet is a lightweight deep learning model designed for fast and efficient image classification, especially on mobile and low-resource devices. 
It uses depthwise separable convolutions, which split standard convolution into two simpler steps, greatly reducing computation and model size. 
This makes MobileNet much faster than traditional CNNs while maintaining good accuracy.
Because of its efficiency, MobileNet is widely used in real-time applications like mobile vision, embedded systems, and edge AI.

<img width="2756" height="2141" alt="Mobilenet image" src="https://github.com/user-attachments/assets/f8895117-4ffd-4109-98de-4477b8b59686" />


MobileNetV2 from torchvision was chosen due to its lightweight design, ease of training, and strong performance on limited datasets.

    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from sklearn.metrics import classification_report
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models,transforms
    import numpy as np
    from PIL import Image
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
    T = 2.5
    
    # Loading the Torchvision model
    
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)

Loading the images to the dataloaders and transforming it according to the MobilenetV2 reqiurements where weights.transforms() alone has no augmentation.

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        weights.transforms()
    ])
    
    val_transform  = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        weights.transforms()])
    
    test_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        weights.transforms()])
    
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    test_ds  = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    
    # class_names = train_ds.classes
    # NUM_CLASSES = len(class_names)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

Replacing the classifier ,Freeze nad unfreeze the layers accordingly.

    # # Replace classifier
    # model.classifier[1] = nn.Linear(
    #     model.classifier[1].in_features,
    #     NUM_CLASSES
    # )
    
    
    # FREEZE ALL layers
    # for param in model.parameters():
    #     param.requires_grad = False
    
    
    # UNFREEZE classifiers
    # for param in model.classifier[1].parameters():
    #     param.requires_grad = True

Layers are all set so here comes the training phase of the model.

    #Training setup
    # model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=LR)
    
    # Training loop with fc
     # for epoch in range(EPOCHS):
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
    # # print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}")

The result is

Epoch 5/5, Train Loss: 0.0721

Next comes the validation phase where energy threshold is computed for dettecting hte unknown images as well.

    # # # # #  ---------- Validation ----------
    # model.eval()
    # val_loss, correct, total = 0, 0, 0
    # all_energy = []
    # with torch.no_grad():
    #         for imgs, labels in val_loader:
    #             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
    #             outputs = model(imgs)
    #             loss = criterion(outputs, labels)
    #             val_loss += loss.item()
    #             energy = energy_score(outputs, T=T)
    #             all_energy.append(energy)
    #             preds = outputs.argmax(dim=1)
    #             correct += (preds == labels).sum().item()
    #             total += labels.size(0)
    
    # val_energy = torch.cat(all_energy)
    # mean_energy = val_energy.mean().item()
    # std_energy = val_energy.std().item()
    # energy_threshold = mean_energy + 2 * std_energy
    # print("Energy Threshold:", energy_threshold)
    # print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct/total:.4f}")

The results are 

Energy Threshold = -2.6104

Epoch 5/5, Train Loss: 0.0721, Val Loss: 0.0608, Val Accuracy: 98.8095

Some known samples were initially classified as unknown due to strict energy thresholds.
This was resolved by defining the threshold using the validation energy distribution (mean + 2σ), ensuring stable open-set recognition.

Here comes saving the model and loading it to test the train and test accuarcy with it.

    # torch.save({
    #     "model_state": model.state_dict(),
    #     "class_names": class_names,
    #     "num_classes": NUM_CLASSES
    # }, "models/mobilenetv2_best.pkl")
    
    # checkpoint = torch.load(
    #     "models/mobilenetv2_best.pkl",
    #     map_location=DEVICE,weights_only=False
    #     )
    
    # model = mobilenet_v2(weights=None)   # weights=None is IMPORTANT
    # model.classifier[1] = torch.nn.Linear(
    #     model.classifier[1].in_features,
    #     checkpoint["num_classes"]
    # )
    # model.load_state_dict(checkpoint["model_state"])
    #class_names = checkpoint["class_names"]
    # model.eval()
    
For torchvision-based models, the architecture was re-instantiated before loading the saved parameters, as PyTorch state dictionaries do not store model structure.

The model features and classifiers are shown below:

MobileNetV2(
  (features): Sequential(
    (0): Conv2dNormActivation(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)      
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)      
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)   
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)   
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)   
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)   
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)   
          (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)   
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)   
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)   
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)   
          (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)   
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)   
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)   
          (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)   
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (16): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)   
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (17): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)   
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (18): Conv2dNormActivation(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=11, bias=True)
  )
)

In order to check the Train accuracy with model.eval() mode where checkpoint model is called by above lines.

    # To check train accuracy
    # model.eval()
    # # print(model)
    # correct, total = 0, 0
    # # correct, total = 0, 0
    
    # with torch.no_grad():
    #     for x, y in train_loader:
    #         x, y = x.to(DEVICE), y.to(DEVICE)
    #         preds = model(x).argmax(dim=1)
    #         correct += (preds == y).sum().item()
    #         total += y.size(0)
    
    # print(f"Train accuracy: {100 * correct / total:.2f}%")

The output is Train accuracy: 99.45%

Same appiles to check the Test accuracy.

    # # # # # Test the model
    # correct, total,test_acc = 0, 0, 0
    # y_true = []
    # y_pred = []
    # with torch.no_grad():
    #    for imgs, labels in test_loader:
    #         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
    #         logits = model(imgs)
    #         # energy = energy_score(logits, T=T)
    
    #         preds = logits.argmax(dim=1)
    #       
    #         correct += (preds == labels).sum().item()
    #         total += labels.size(0)
    #         y_true.extend(labels.cpu().numpy())
    #         y_pred.extend(preds.cpu().numpy())
    #         test_acc = correct / total * 100
    
    # print(f"Test Accuracy: {test_acc:.2f}%")

The output of this batch is Test Accuracy: 99.40%.

The model attained 99.45% training accuracy and 99.40% testing accuracy, showing excellent generalization and stable performance.

The Classification report adn Confusion matrix are generated thorugh the codes given below:

    # # classification report
    # print(classification_report(
    #     y_true,
    #     y_pred,
    #     target_names=class_names   # list of 11 class names
    # ))
    
    # #Confusion matrix can be added similarly
    
    # cm = confusion_matrix(y_true, y_pred)
    
    # plt.figure(figsize=(8,6))
    # sns.heatmap(cm, annot=True, fmt="d",
    #             xticklabels=class_names,
    #             yticklabels=class_names)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.show()

The results of those reports are 

    ***Classification Report****
                                       precision    recall  f1-score   support
                                       
                         animal fish       0.98      0.99      0.99       520
                    animal fish bass       0.80      0.31      0.44        13
       fish sea_food black_sea_sprat       0.99      0.99      0.99       298
       fish sea_food gilt_head_bream       1.00      1.00      1.00       305
       fish sea_food hourse_mackerel       0.99      1.00      1.00       286
            fish sea_food red_mullet       1.00      1.00      1.00       291
         fish sea_food red_sea_bream       1.00      1.00      1.00       273
              fish sea_food sea_bass       1.00      0.99      1.00       327
                fish sea_food shrimp       1.00      1.00      1.00       289
    fish sea_food striped_red_mullet       1.00      1.00      1.00       293
                 fish sea_food trout       0.99      1.00      0.99       292
    
                            accuracy                           0.99      3187
                           macro avg       0.98      0.93      0.95      3187
                        weighted avg       0.99      0.99      0.99      3187

<img width="1280" height="612" alt="Confusion matrix mobilenetv2" src="https://github.com/user-attachments/assets/c723ae4f-c5ec-46c2-bf4a-7963fae17f52" />

Finally comes the single image inference codes.

    # Single Image Inference
    image_path = "D:/Multiclass Fish Image classification/images.cv/data/SHRIMP.jpg"
    Energy_T = -2.6104
    model.eval()
    with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            image_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
            logits = model(image_tensor)
            energy = -torch.logsumexp(logits, dim=1).item()
            pred_idx = logits.argmax(dim=1).item()
            conf = torch.softmax(logits/T, dim=1).max().item()
    
    if  energy > Energy_T and conf <= 0.99:
            print(f"Unknown image detected | Energy: {energy:.2f} | Confidence: {conf * 100:.2f}")
    else:
            print(f"Predicted: {class_names[pred_idx]} | Energy: {energy:.2f} | Confidence: {conf * 100:.2f}")

The outputs of the model with different images are given below:

Predicted: fish sea_food trout | Energy: -4.01 | Confidence: 72.13

Unknown image detected | Energy: -2.05 | Confidence: 57.36

Predicted: fish sea_food shrimp | Energy: -4.15 | Confidence: 68.03

With 99.45% training accuracy and 99.40% testing accuracy, the model accurately classified known images and effectively identified unknown samples using an energy-based approach.

Limitations: 

The model works well on the given dataset but may struggle when trained on limited data or images with complex backgrounds. 
The method used to detect unknown images depends on a fixed threshold, which may not always give perfect results.

Future Scope: 

In the future, better data augmentation and improved unknown detection methods can be used to make the model more robust and reliable on new and unseen images





    
