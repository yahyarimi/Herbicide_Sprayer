# AI-Based Precision Weed Herbicide Spraying Robot

## Overview
The AI-based precision weed herbicide spraying robot is an intelligent agricultural system designed to detect and selectively spray weeds. It utilizes a deep learning model built with TensorFlow to identify weeds in real time and control a spraying mechanism accordingly. By precisely targeting weeds, this system helps reduce herbicide waste, lower costs, and minimize environmental impact.

## Features
- **Real-time Weed Detection:** Uses a deep learning model to classify weeds from crops.
- **Selective Spraying:** Herbicide is sprayed only on detected weeds, reducing chemical usage.
- **Raspberry Pi 3 Deployment:** The model runs on an embedded system, enabling field applications.
- **Camera-based Vision System:** Captures images for analysis and decision-making.
- **Autonomous Operation:** Can be integrated with robotic platforms for automated farming.

## Hardware Requirements
- Raspberry Pi 3 Model B/B+
- Raspberry Pi Camera Module
- Servo-controlled Spraying Mechanism
- Battery Pack (suitable for Raspberry Pi)
- SD Card (minimum 16GB, recommended 32GB)

## Software Requirements
- Python 3.7+
- TensorFlow (Lite version for Raspberry Pi)
- OpenCV
- NumPy
- RPi.GPIO (for controlling spraying mechanism)
- Picamera (for capturing images)

## Model Training & Deployment
### Model Training
- The weed detection model was trained using a dataset of crop and weed images.
- A Convolutional Neural Network (CNN) architecture was used.
- The model was trained using TensorFlow and optimized for real-time inference.

### Deployment on Raspberry Pi
- The trained model was converted to TensorFlow Lite format for efficient execution.
- Image processing and inference were handled using OpenCV and TensorFlow Lite.
- The model was integrated with GPIO controls to trigger the spraying mechanism.

## Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/weed-spraying-robot.git
   cd weed-spraying-robot
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python numpy RPi.GPIO picamera
   ```
3. Run the detection and spraying script:
   ```bash
   python main.py
   ```

## Usage Instructions
- Ensure the camera is properly positioned for capturing images.
- Run the script to start real-time weed detection.
- The system will trigger the sprayer when a weed is detected.

## Example Output
- Example images of detected weeds and successful spraying.
- Log output showing detection confidence and spray trigger status.

## Troubleshooting
- **Model not loading?** Ensure TensorFlow Lite is installed correctly.
- **Camera not working?** Check if `picamera` is enabled on Raspberry Pi.
- **Sprayer not activating?** Verify GPIO pin connections.

## Contributors & Credits
- Developed by Yahya Abdurrazaq
- Special thanks to open-source datasets and TensorFlow community

## License
This project is licensed under the MIT License.

