Image Classification is a core computer vision task that involves assigning a predefined label to an image based on its visual content. 
This project focuses on classifying fish images into multiple categories using deep learning models. 
The task involves training a CNN from scratch and leveraging transfer learning with pre-trained models to enhance performance. 
The project also includes saving models for later use and deploying a Streamlit application to predict fish categories from user-uploaded images.

Customised CNN was built from scratch to focus on classifying the fish images into multiple categories and evaluated its performance.
Experimented with 5 different pre-trained models for this classification tasks.Models are VGG16,ResNET-50,MobileNetV2,InceptionV3,Vision Transformer(ViT-base16).

| Model              | Main Idea                       | Depth          | Parameters | Advantage     | Limitation                                                 |
| ------------------ | ------------------------------- | -------------- | ---------- | ----------------------------------------------- | ---------------------------------------------------------- |
| CNN                | Basic convolution layers        | Shallow–Medium | Medium     | Simple and effective                            | Not suitable for very deep networks                        |
| ResNet50           | Residual skip connections       | Very Deep      | Medium     | Solves vanishing gradient problem               | Slightly complex architecture                              |
| VGG16              | Uniform 3×3 convolutions        | Deep           | Very High  | Strong feature extraction capability            | High computational cost                                    |
| InceptionV3        | Parallel multi-scale filters    | Deep           | Moderate   | Efficient multi-scale feature extraction        | Complex architecture                                       |
| MobileNetV2        | Depthwise separable convolution | Medium         | Low        | Lightweight and fast                            | Slightly lower accuracy                                    |
| Vision Transformer | Self-attention on image patches | Deep           | High       | Captures global image relationships effectively | Requires large datasets and higher computational resources |

