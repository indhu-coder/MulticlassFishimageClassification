Why ViT when we already have CNNs?

CNNs:

-->Use convolutions → local feature learning

-->Strong inductive bias (edges, textures)

-->Work well with smaller datasets

ViT:

-->Uses self-attention

-->Learns global relationships directly

-->Scales better with large datasets

The Simple Architecture Summary of ViT-base-patch 16-224 of Google's from Hugging Face platform

Image
 ↓
Split into patches
 ↓
Linear embedding + position embedding
 ↓
Transformer Encoder (L layers)
 ↓
[CLS] token
 ↓
MLP Head
 ↓
Class prediction

Image is shown below:

<img width="685" height="357" alt="ViT image" src="https://github.com/user-attachments/assets/ea5a9c10-8438-49ba-b828-03f642e148d2" />

Advantages of ViT:

✅ Captures global context
✅ Scales well with large datasets
✅ Simple architecture (no convs)
✅ Works well with transfer learning

Limitations:

❌ Needs large datasets (e.g. ImageNet-21k)
❌ Slower to train from scratch
❌ Less effective on small datasets without pretraining

Pre-trained model has been taken from Hugging face platform.Imporing the modules and loading images from the folders are given below:

    from transformers import ViTForImageClassification, ViTImageProcessor
    from PIL import Image
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, models,transforms
    
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TRAIN_DIR = "D:/Multiclass Fish Image classification/images.cv/data/train"
    VAL_DIR   = "D:/Multiclass Fish Image classification/images.cv/data/val"
    TEST_DIR  = "D:/Multiclass Fish Image classification/images.cv/data/test"
    
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3

Loading the model, dataloaders and preprocessing the image for this particular model are given below:

    # Load pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,ignore_mismatched_sizes=True
    )
    
    # Image processor for ViT
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )
    
    # Datasets with ViT processor
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=processor)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=processor)
    test_ds  = datasets.ImageFolder(TEST_DIR, transform=processor)
    
    class_names = train_ds.classes
    NUM_CLASSES = len(class_names)

While preprocessing the image following limitations were popped up so used function accordingly.

-->HF processor returns BatchFeature

-->Dataset wraps it in a tuple with label

-->Default DataLoader collates into lists

➡️ collate_fn is the only correct fix

    def collate_fn(batch):
        pixel_values = torch.stack([
            torch.as_tensor(item[0]["pixel_values"][0])  # <-- take tensor from list
            for item in batch
        ])

    labels = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long
    )
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

  ViT model features and classifiers are given below for modification of layers in later stages.

  ViTModel(
  (embeddings): ViTEmbeddings(
    (patch_embeddings): ViTPatchEmbeddings(
      (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (dropout): Dropout(p=0.0, inplace=False)
  )
  (encoder): ViTEncoder(
    (layer): ModuleList(
      (0-11): 12 x ViTLayer(
        (attention): ViTAttention(
          (attention): ViTSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
          )
          (output): ViTSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        (intermediate): ViTIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): ViTOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
    )
  )
  (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  (pooler): ViTPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)


Modification of classifier head ,freezing and Unfreezing the layers with the training setup (Incl optimizer)are given below:

    # Modify the classifier head
    model.classifier = nn.Linear(
        in_features=model.classifier.in_features,
        out_features=NUM_CLASSES
    )
    
    
    # Freeze all ViT layers
    for param in model.vit.parameters():
        param.requires_grad = False
    
    #unfreeze classifier layers
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Training setup
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim. AdamW(
        model.classifier.parameters(),
        lr=LR
    )

  ✔️ This freezes:

--->Patch embedding

--->Transformer encoder

--->LayerNorms

📌 Why AdamW for the optimizer?

--->Decoupled weight decay

--->Stable for Transformers

--->Used in almost all ViT papers

Next comes the Training phase.

    # Training loop with ViT
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        # model.train()

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total * 100

    # print(
    #     f"Epoch [{epoch+1}/{EPOCHS}] "
    #     f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%"
    # )

    
Train loss and accuracy during training:

<img width="1280" height="612" alt="Training Accuracy" src="https://github.com/user-attachments/assets/fdc0e998-a099-465c-93f7-31f19d1018a1" />

<img width="640" height="480" alt="Training loss" src="https://github.com/user-attachments/assets/d5201a22-a49b-4722-b286-5139a76cfcbc" />



This tells us:

✅ Classifier head is learning fast

✅ Backbone features are very strong

✅ Loss decreasing smoothly (no instability)

✅ No exploding / vanishing gradients

✅ Data pipeline is now correct

Validation phase starts here.

    #Validation loop can be added similarly
    model.eval()
    correct, total,val_loss = 0, 0,0
    with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

        
            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            val_loss +=loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_acc = correct / total * 100
        # print(f"Epoch {epoch+1}/{EPOCHS} | "
        #   f"Val Loss: {val_loss/len(val_loader):.4f} | "
        #   f"Val Acc: {val_acc:.2f}%")
    
The result is

Epoch 5/5 | Val Loss: 0.0336 | Val Acc: 99.27%

Now its time to check the Training accuracy and testing accuracy with the saved model checkpoints.

    # Save the best model
    # model.save_pretrained("models/vit_11cls")
    # processor.save_pretrained("models/vit_11cls")
    
    # torch.save(class_names, "models/class_names.pt")
  
    
 Model Configuration  after saving it given below:

     ViTConfig {
      "architectures": [
        "ViTForImageClassification"
      ],
      "attention_probs_dropout_prob": 0.0,
      "dtype": "float32",
      "encoder_stride": 16,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.0,
      "hidden_size": 768,
      "id2label": {
        "0": "animal fish",
        "1": "animal fish bass",
        "2": "fish sea_food black_sea_sprat",
        "3": "fish sea_food gilt_head_bream",
        "4": "fish sea_food hourse_mackerel",
        "5": "fish sea_food red_mullet",
        "6": "fish sea_food red_sea_bream",
        "7": "fish sea_food sea_bass",
        "8": "fish sea_food shrimp",
        "9": "fish sea_food striped_red_mullet",
        "10": "fish sea_food trout"
      },
      "image_size": 224,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "label2id": {
        "animal fish": 0,
        "animal fish bass": 1,
        "fish sea_food black_sea_sprat": 2,
        "fish sea_food gilt_head_bream": 3,
        "fish sea_food hourse_mackerel": 4,
        "fish sea_food red_mullet": 5,
        "fish sea_food red_sea_bream": 6,
        "fish sea_food sea_bass": 7,
        "fish sea_food shrimp": 8,
        "fish sea_food striped_red_mullet": 9,
        "fish sea_food trout": 10
      },
      "layer_norm_eps": 1e-12,
      "model_type": "vit",
      "num_attention_heads": 12,
      "num_channels": 3,
      "num_hidden_layers": 12,
      "patch_size": 16,
      "pooler_act": "tanh",
      "pooler_output_size": 768,
      "qkv_bias": true,
      "transformers_version": "4.57.3"
    }

Loading the saved model to Train dataloader with model.eval() to chek the Train and Test accuracy.

      # # Loading the saved model
    model = ViTForImageClassification.from_pretrained("models/vit_11cls_V3")
    model.to(DEVICE)
    processor = ViTImageProcessor.from_pretrained("models/vit_11cls_V3")
    class_names = torch.load("models/class_names_V3.pt")
        # # To check train accuracy with the saved model checkpoints
        # model.eval()
        # # print(model)
        # correct, total = 0, 0
        # with torch.no_grad():
        #     for batch in train_loader:
        #         pixel_values = batch["pixel_values"].to(DEVICE)
        #         labels = batch["labels"].to(DEVICE)
        #         outputs = model(
        #             pixel_values=pixel_values,
        #             labels=labels
        #         )
        
    #         logits = outputs.logits
    #         preds = torch.argmax(logits, dim=1)
    #         correct += (preds == labels).sum().item()
    #         total += labels.size(0)
    #     train_acc = correct / total * 100
    #     print(f"Train Accuracy: {train_acc:.2f}%")

    # To find the Test accuracy with saved model checkpoints
    correct, total = 0, 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
    test_acc = correct / total * 100
    print(f"Test Accuracy: {test_acc:.2f}%")
The results are as follows:

Train Accuracy: 99.71%

Test Accuracy: 99.31%

To find the classification report:-

    # classification report
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names   # list of 11 class names
    ))

******Classification report*****

                                      precision    recall  f1-score   support
    
                         animal fish       0.98      0.98      0.98       520
                    animal fish bass       0.80      0.62      0.70        13
       fish sea_food black_sea_sprat       1.00      1.00      1.00       298
       fish sea_food gilt_head_bream       0.99      1.00      0.99       305
       fish sea_food hourse_mackerel       1.00      1.00      1.00       286
            fish sea_food red_mullet       0.99      1.00      0.99       291
         fish sea_food red_sea_bream       1.00      0.99      1.00       273
              fish sea_food sea_bass       0.99      1.00      1.00       327
                fish sea_food shrimp       1.00      1.00      1.00       289
    fish sea_food striped_red_mullet       1.00      1.00      1.00       293
                 fish sea_food trout       1.00      0.99      1.00       292
    
                            accuracy                           0.99      3187
                           macro avg       0.98      0.96      0.97      3187
                        weighted avg       0.99      0.99      0.99      3187

Followed by Confusion matrix:

    #Confusion matrix can be added similarly
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

<img width="1280" height="612" alt="confusion matrix ViT" src="https://github.com/user-attachments/assets/4756f55c-6ae4-4598-8624-78f7a4ba2499" />

Finally single image inference with Energy based detection where Energy is chosed from the validation set.

For known vs unknown, we’ll use energy score:

      Energy(x)=−T⋅logi∑​eTlogitsi
      ​​
Low energy → confident → known

High energy → uncertain → unknown

Threshold is chosen from validation set.

    def energy_score(logits, T=1.0):
        return -T * torch.logsumexp(logits / T, dim=1)

Then the validation phase has beed changed slightly in order to find the energy with the saved model checkpoint.

     for epoch in range(EPOCHS):
    #     model.eval()
    #     correct, total,val_loss = 0, 0,0
    # with torch.no_grad():
    #         for batch in val_loader:
    #             pixel_values = batch["pixel_values"].to(DEVICE)
    #             labels = batch["labels"].to(DEVICE)
    #             outputs = model(
    #                 pixel_values=pixel_values,
    #                 labels=labels)
    #             loss = outputs.loss
    #             logits = outputs.logits
    #             val_loss +=loss.item()
    #             preds = torch.argmax(logits, dim=1)
    #             correct += (preds == labels).sum().item()
    #             total += labels.size(0)
    #             val_acc = correct / total * 100
    #             energy = energy_score(logits).mean().item()
    #             ENERGY_T = -torch.logsumexp(logits / T, dim=1).mean().item()  # Example threshold based on confidence
    #         print(f"Epoch {epoch+1}/{EPOCHS} | "
    #              f" Energy: {energy:.4f} | ")
    # print ("Chosen Energy Threshold: ", ENERGY_T)
    
The chosen energy is 

Epoch 3/3 |  Energy: -5.7196 | 

Chosen Energy Threshold:  -2.5218210220336914

Snippet for single image inference is as follows:

    def predict_image(image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(DEVICE)
    
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            confidence = torch.softmax(logits, dim=1).max().item()
            energy = -torch.logsumexp(logits / T, dim=1).item()
          
    
            pred = logits.argmax(dim=1).item()
    
            if energy > ENERGY_T and confidence <= 0.9995:
                class_name = "unknown"
            else:
                class_name = class_names[pred]
    
        return class_name, confidence, energy, 
    
    image_path = "D:/Multiclass Fish Image classification/images.cv/data/9.jpg"
    
    class_name, confidence, energy = predict_image(image_path)
    print(f"Predicted Class : {class_name}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print(f"Energy          : {energy:.4f}")

Finally the predictions are:-

Predicted Class : fish sea_food trout
Confidence      : 98.94%
Energy          : -2.5912

Predicted Class : fish sea_food shrimp
Confidence      : 99.96%
Energy          : -3.1411

Predicted Class : unknown
Confidence      : 26.99%
Energy          : -2.3378

Limitations and Future Scope:-

Vision Transformers show high accuracy for image classification tasks; however, they have certain limitations. 
The model requires a large amount of training data and is sensitive to background variations in images. 
In addition, Vision Transformers do not naturally support the detection of unknown classes.
Future improvements can include the use of advanced architectures, stronger data augmentation techniques to improve generalization, and improved methods for unknown class detection. 
Furthermore, fine-tuning deeper layers of the model and addressing class imbalance can enhance the robustness and overall performance of the system.



