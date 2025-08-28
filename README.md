# Facial Recognition System in Python

This project implements a **real-time facial recognition system** using Convolutional Neural Networks (CNNs), OpenCV, TensorFlow/Keras, and Albumentations.  
The system was trained on a custom dataset of images taken of our group (3 participants). Each image was annotated with bounding boxes and labels (`JayFace`, etc.) using **LabelMe**, then augmented to improve robustness.  
The final trained model detects and recognizes faces in live webcam video, drawing bounding boxes around each face and displaying the person’s name.

---

## 🚀 Features
- Custom dataset (~3250 images) created from pictures of our team  
- Bounding box annotations with **LabelMe**  
- Dataset augmentation (flipping, cropping, brightness, etc.) via **Albumentations**  
- CNN architecture based on pretrained **VGG16**  
- Dual-branch network for **classification** and **bounding box regression**  
- Real-time detection via **OpenCV** with labels displayed above faces  
- Achieved **93%+ accuracy, precision, recall, and F1-score** :contentReference[oaicite:1]{index=1}

---

## 📸 Visuals / Demo
### Example Workflow:
1. **Annotated Dataset with Labels**  
   
<img width="74" height="84" alt="Screenshot 2025-08-28 at 2 25 55 PM" src="https://github.com/user-attachments/assets/ed16011a-e33d-4ef9-b45b-9297c9dc3184" />


2. **Sample Predictions on Test Data**  
   ![Prediction Example](docs/prediction_example.png)

3. **Real-Time Recognition**  
   ![Realtime Example](docs/realtime_example.png)
