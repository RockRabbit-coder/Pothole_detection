import cv2
import csv
import time
import os
import numpy as np
import tensorflow as tf

# ================= CONFIG =================
MODEL_PATH = "best_int8.tflite"
SAVE_DIR = "detected_frames_camera"
LOG_FILE = "pothole_log_camera.csv"

CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.45
NUM_THREADS = 4
FRAME_SKIP = 1
# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)

# -------- Load Model --------
interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=NUM_THREADS
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

INPUT_HEIGHT = input_shape[1]
INPUT_WIDTH = input_shape[2]

is_quantized = input_dtype == np.uint8
out_scale, out_zero = output_details[0]["quantization"]

print("Model input size:", INPUT_WIDTH, "x", INPUT_HEIGHT)
print("Quantized:", is_quantized)

# -------- Open Camera --------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fps_video = 30  # assumed FPS
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------- CSV Setup --------
csv_file = open(LOG_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_id",
    "timestamp_sec",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "image_path"
])

frame_counter = 0
processed_counter = 0
detected_counter = 0
start_time = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % FRAME_SKIP != 0:
        continue

    processed_counter += 1

    # -------- Preprocess --------
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

    if is_quantized:
        input_data = resized.astype(np.uint8)
    else:
        input_data = resized.astype(np.float32) / 255.0

    input_data = np.expand_dims(input_data, axis=0)

    # -------- Inference --------
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

    # -------- Dequantize --------
    if is_quantized:
        output = (raw_output.astype(np.float32) - out_zero) * out_scale
    else:
        output = raw_output

    if output.shape[0] == 5:
        output = output.T

    boxes = []
    scores = []

    # -------- Decode YOLO --------
    for pred in output:
        if len(pred) < 5:
            continue

        xc, yc, bw, bh, conf = pred[:5]

        if conf < CONF_THRESHOLD:
            continue

        x = int((xc - bw / 2) * W)
        y = int((yc - bh / 2) * H)
        w_box = int(bw * W)
        h_box = int(bh * H)

        boxes.append([x, y, w_box, h_box])
        scores.append(float(conf))

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                conf = scores[i]

                x = max(0, x)
                y = max(0, y)
                w_box = min(w_box, W - x)
                h_box = min(h_box, H - y)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w_box, y + h_box),
                    (0, 255, 0),
                    2
                )

                label = f"Pothole {conf:.2f}"

                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                text_x = x
                text_y = y - 10

                if text_y - text_h < 0:
                    text_y = y + text_h + 10

                if text_x + text_w > W:
                    text_x = W - text_w - 5

                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_h - baseline),
                    (text_x + text_w, text_y + baseline),
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

            # Save frame
            image_name = f"frame_{frame_counter}.jpg"
            image_path = os.path.join(SAVE_DIR, image_name)
            cv2.imwrite(image_path, frame)

            timestamp = frame_counter / fps_video
            i = indices.flatten()[0]
            x, y, w_box, h_box = boxes[i]
            conf = scores[i]

            csv_writer.writerow([
                frame_counter,
                round(timestamp, 2),
                round(conf, 3),
                x, y, w_box, h_box,
                image_path
            ])

            detected_counter += 1

    # Show Live Feed
    cv2.imshow("Live Pothole Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # FPS Monitor
    if processed_counter % 20 == 0:
        elapsed = time.time() - start_time
        fps_live = processed_counter / elapsed
        print(f"Processing FPS: {fps_live:.2f} | Detections: {detected_counter}")

# -------- Cleanup --------
cap.release()
cv2.destroyAllWindows()
csv_file.close()

total_time = time.time() - start_time

print("\n✅ Finished")
print("Total Frames Read:", frame_counter)
print("Frames Processed:", processed_counter)
print("Average Processing FPS:", processed_counter / total_time)
print("Total Detections:", detected_counter)
