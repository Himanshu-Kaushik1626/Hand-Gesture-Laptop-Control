# ✋ Virtual Mouse – Hand Gesture Control

Control your laptop using hand gestures in real-time using computer vision. This project replaces traditional input devices like a mouse with intuitive hand movements captured via webcam.

---

## 🎥 Demo

<!-- Add your demo GIF or video here -->

![Demo](demo.gif)
<img width="1920" height="1080" alt="Screenshot 2026-03-27 213601" src="https://github.com/user-attachments/assets/29b6faa5-5707-4381-8b00-e61bb14a231f" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/65139a67-8710-40c9-9460-ad3b1aa792b7" />

---

## 🚀 Features

* 🖱 Cursor control using hand movements
* 👆 Left & Right click gestures
* 🔄 Scrolling functionality
* ✊ Drag and drop support
* ⚡ Real-time performance

---

## 🛠 Tech Stack

* Python
* OpenCV
* MediaPipe
* PyAutoGUI
* NumPy

---

## 🎮 Gesture Controls

| Gesture                 | Action      |
| ----------------------- | ----------- |
| ☝ Index Finger          | Move Cursor |
| 🤏 Index + Thumb Pinch  | Left Click  |
| 🤘 Middle + Thumb Pinch | Right Click |
| ✌ Index + Middle        | Scroll      |
| ✊ Fist                  | Drag        |

---

## 📂 Project Structure

```
Hand-Gesture-Laptop-Control/
│── main.py              # Entry point
│── mouse_controller.py  # Gesture → Mouse logic
│── hand_tracking.py     # Hand detection (MediaPipe)
│── ui.py                # UI (CustomTkinter)
│── requirements.txt     # Dependencies
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/Hand-Gesture-Laptop-Control.git
cd Hand-Gesture-Laptop-Control
```

### 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```
python main.py
```

---

## 💡 Motivation

This project demonstrates how computer vision can be used to create touchless human-computer interaction systems, making technology more accessible and futuristic.

---

## ⚡ Challenges

* Maintaining accuracy in different lighting conditions
* Reducing cursor jitter
* Reliable gesture detection

---

## 🔮 Future Improvements

* Custom gesture mapping
* Improved accuracy using ML models
* Mobile device integration
* Voice + gesture hybrid control

---

## 🤝 Contributing

Feel free to fork this repo and improve it. Pull requests are welcome!

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
