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
