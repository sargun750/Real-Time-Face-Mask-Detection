# Real-Time Face Mask Detector

Real-time face mask detector built in TensorFlow/Keras using MobileNetV2 with a custom binary classification head.

- Fine-tuned via transfer learning on the RMFD dataset (90K images), reaching **99.9% validation accuracy**.
- Integrated OpenCV's ResNet-10 SSD for face detection, achieving **150 ms end-to-end latency on CPU**.

## Idea

Mask detection is split into two simpler problems:

1. **Where are the faces?** A pretrained OpenCV ResNet-10 SSD face detector answers this quickly on a CPU.
2. **Is this face masked?** A small binary classifier looks only at the cropped face.

For the classifier, training a CNN from scratch on 90K images would be slow and unnecessary. Instead, MobileNetV2 pretrained on ImageNet supplies the visual features, and only a small head is trained on top. MobileNetV2 is built for low compute, which keeps webcam inference real-time without a GPU.

```
Webcam frame -> ResNet-10 SSD (face boxes) -> crop + resize 224x224 + normalize -> MobileNetV2 classifier -> Mask / No Mask
```

## How the three scripts work

### 1. `Split_Images.py` - prepare the dataset
- Recursively walks the RMFD folders `AFDB_masked_face_dataset` (masked) and `AFDB_face_dataset` (unmasked), collecting every `.jpg`, `.jpeg` and `.png`.
- Does an **80/20 train/validation split** with `train_test_split(random_state=42)`, so the split is reproducible.
- Copies the files into `dataset/train/{mask,no_mask}` and `dataset/val/{mask,no_mask}`, the folder layout Keras' `flow_from_directory` expects.
- Raises an error if a source folder has no images, so a wrong path fails loudly.
- Set `extracted_dir` at the top of the script to your RMFD folder. Create the four output folders first, because `shutil.copy` won't create them.

### 2. `Train_Model.py` - train the classifier
- **Model:** MobileNetV2 (ImageNet weights, no top, 224x224x3 input) with **all base layers frozen**, followed by `GlobalAveragePooling2D`, `Dense(128, relu)` and `Dense(1, sigmoid)`.
- **Data:** the training generator rescales pixels to 0-1 and augments with rotation (up to 20 degrees), zoom (up to 20%) and horizontal flips, which helps the model cope with varied webcam conditions. The validation generator only rescales. Batch size is 32.
- **Training:** Adam (lr = 1e-4) with binary cross-entropy, for up to 10 epochs.
- **Callbacks:** `EarlyStopping` on `val_loss` (patience 3, restores the best weights) and `ModelCheckpoint` saving the best model to `mask_model.h5`.
- **Output:** evaluates on the validation set and saves `mask_model_final.h5`. A full run took about 3 hours; training was stopped around epoch 7 once accuracy plateaued near 99.9%.
- **Label mapping:** Keras assigns class indices alphabetically (`mask` = 0, `no_mask` = 1), so the sigmoid output is the **probability of "No Mask"**.

### 3. `Mask Project.py` - real-time inference
- Loads `mask_model_final.h5` and the Caffe face detector (`deploy.prototxt.txt` + `res10_300x300_ssd_iter_140000.caffemodel`).
- For each webcam frame, `detect_faces_dnn` resizes it to 300x300, builds a blob with the detector's mean values (104, 177, 123), and keeps detections with **confidence > 0.5**. Box coordinates are scaled back to the original frame size.
- Each face crop is resized to 224x224, divided by 255 (matching training) and given a batch dimension.
- If the model output is **> 0.5, the face is labelled "No Mask" (red box)**, otherwise "Mask Detected" (green box). The detector's confidence is shown next to the label.
- The result is shown live in an OpenCV window. Press **`q`** to quit.

## Applications

- Compliance monitoring in hospitals, labs, clinics and workplaces
- Entry-gate and kiosk access control (admit only when masked)
- Public-space and transport-hub safety monitoring
- Lightweight edge or embedded deployments, since it runs on a CPU

## Install

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## Run

```bash
python Split_Images.py     # 1. organise RMFD into dataset/train and dataset/val
python Train_Model.py      # 2. train and save mask_model_final.h5
python "Mask Project.py"   # 3. start the webcam detector
```

`Mask Project.py` has hardcoded `deploy` and `model_path` values. Change them to wherever the detector files are on your machine.
