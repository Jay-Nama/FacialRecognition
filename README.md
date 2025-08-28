# Facial Recognition System in Python

This project implements a **real-time facial recognition system** using Convolutional Neural Networks (CNNs), OpenCV, TensorFlow/Keras, and Albumentations.  
The system was trained on a custom dataset of images taken of our group (3 participants). Each image was annotated with bounding boxes and labels (`JayFace`, etc.) using **LabelMe**, then augmented to improve robustness.  
The final trained model detects and recognizes faces in live webcam video, drawing bounding boxes around each face and displaying the person’s name.

Note: The dataset (pictures + LabelMe annotations) is private and not included in this repository. The scripts will not run directly without creating your own dataset. Instead, this repo demonstrates the **workflow, code, and methodology** behind our project.

---

## 🚀 Features
- Custom dataset (~3250 images) created from pictures of our team  
- Bounding box annotations with **LabelMe**  
- Dataset augmentation (flipping, cropping, brightness, etc.) via **Albumentations**  
- CNN architecture based on pretrained **VGG16**  
- Dual-branch network for **classification** and **bounding box regression**  
- Real-time detection via **OpenCV** with labels displayed above faces  
- Achieved **93% accuracy, 94% precision, 95% recall, and 95% F1-score**

---

## 📸 Visuals / Demo
### Example Workflow:
Note: Images were used in all kinds of settings like dim lighs, bright lights, face fully visible, face fully cut off, face half cut off, etc.

1. **Annotated Dataset with Labels**  
Example:  
<img width="72" height="76" alt="Screenshot 2025-08-28 at 2 28 48 PM" src="https://github.com/user-attachments/assets/b10398ef-d04d-4736-9cd1-892ef734a11f" />

2. **Sample Predictions on Test Data**  
Example:
<img width="74" height="84" alt="Screenshot 2025-08-28 at 2 25 55 PM" src="https://github.com/user-attachments/assets/ed16011a-e33d-4ef9-b45b-9297c9dc3184" />

3. **Real-Time Recognition**  
Example:
<img width="233" height="122" alt="Screenshot 2025-08-28 at 2 29 20 PM" src="https://github.com/user-attachments/assets/341d5c7e-abd0-4f1f-85fa-ab130405ba01" />

