import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model and allocate tensors
model_path = 'best_yolov5_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to get output tensors
def get_output_tensor(interpreter, index):
    tensor = interpreter.tensor(interpreter.get_output_details()[index]['index'])()
    return np.squeeze(tensor)

# Function to draw bounding boxes
def draw_bounding_boxes(image, boxes, scores, classes, threshold=0.4):
    for i in range(len(scores)):
        if scores[i] > threshold:
            box = boxes[i]
            start_point = (int(box[1] * image.shape[1]), int(box[0] * image.shape[0]))
            end_point = (int(box[3] * image.shape[1]), int(box[2] * image.shape[0]))
            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            label = f"Class: {int(classes[i])}, Score: {scores[i]:.2f}"
            image = cv2.putText(image, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame
    input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the output tensors
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)

    # Print intermediate outputs for debugging
    print(f"Boxes: {boxes}")
    print(f"Classes: {classes}")
    print(f"Scores: {scores}")

    # Draw bounding boxes on the frame
    frame = draw_bounding_boxes(frame, boxes, scores, classes)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
