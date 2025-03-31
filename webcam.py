import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="custom_model_lite\detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels for classification
labels = ['Weed', 'Crop']

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

def draw_bounding_boxes(frame, boxes, classes, scores, threshold=0.5):
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            class_id = int(classes[i])
            label = labels[class_id]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw label and score
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.float32(input_data)  # Ensure the input data type matches the model's expected input

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the model's output
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: [1, num_boxes, 4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Shape: [1, num_boxes]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Shape: [1, num_boxes]

    # Draw bounding boxes on the frame
    draw_bounding_boxes(frame, boxes, classes, scores)

    # Display the frame with the predicted class label
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
