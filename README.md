# ♻️ Smart Sorter
### Made by Arnav Sharma and Siddarth Arunachalam


Smart Sorter is a full-stack web application built at a hackathon to help users properly dispose of waste using image classification. By leveraging machine learning and a modern web stack, the app identifies items from uploaded images and determines whether they belong in **trash, recycling, compost, or other** categories.

The goal of Smart Sorter is to reduce disposal confusion and promote more environmentally responsible habits through an intuitive, user-friendly interface.

---

## 🚀 Features

- **Image-Based Trash Classification**
  - Upload an image of an item and receive an instant disposal recommendation
  - Powered by a TensorFlow image classification model

- **User Accounts & Authentication**
  - Secure account creation and login
  - Personalized experience for each user

- **Disposal History Tracking**
  - Logged-in users can view a history of previously classified items
  - Enables quick reference and learning over time

- **Clean & Responsive UI**
  - Built with HTML and CSS for clarity and ease of use
  - Designed for fast interaction during real-world use

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS

### Backend
- Flask
- SQLAlchemy
- SQLite / PostgreSQL (depending on configuration)

### Machine Learning
- TensorFlow  
  - Image classification model used to predict disposal categories
  - Integrated directly into the Flask backend for inference

---

## 🧠 How It Works

1. A user uploads an image of an item through the web interface.
2. The Flask backend processes the image and sends it to a TensorFlow model.
3. The model predicts the disposal category (Trash, Recycle, Compost, or Other).
4. The result is returned to the frontend and displayed to the user.
5. If the user is logged in, the classification is saved to their history for future reference.

---

## 🧪 Hackathon Context

This project was built collaboratively during a HackUMASS 2025 with a focus on:
- End-to-end full-stack development
- Practical application of machine learning
- Environmental impact and sustainability

Due to time constraints, the model and feature set prioritize clarity and usability over exhaustive classification coverage.

---

## 🔮 Future Improvements

- Improve model accuracy with a larger and more diverse dataset
- Support real-time camera capture
- Expand disposal categories based on local regulations
- Develop a mobile-friendly or native app version

