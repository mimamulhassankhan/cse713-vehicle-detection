# What happens during training and testing?

## The task

For an image, answer two questions: **Where is each vehicle? What kind of vehicle is it?** A rectangle gives its location. A label gives its type.

## Step 1: Read the answers provided by the dataset

Each DETRAC XML file contains frames and their vehicles. A vehicle box is stored as left, top, width, and height. We convert it into YOLO's center X, center Y, width, and height, divided by image dimensions so values are between 0 and 1.

Example: in a 1000 by 500 image, a box starting at (100,100) with width 200 and height 100 has center (200,150). Its YOLO values are `0.2 0.3 0.2 0.2`.

These labels are reference answers. Reading them and drawing boxes is annotation visualization, not automatic detection.

## Step 2: Separate the data

**Training data** teaches the model. **Validation data** helps choose a model and settings. **Testing data** checks the final model on examples reserved until the end.

Neighboring video frames can look almost identical. If one goes into training and the next into testing, the reported result can be misleading. We keep every sequence in just one split. This is not proof of unseen camera locations.

## Step 3: Start with useful visual knowledge

YOLOv8n is a small object-detection neural network. Its pretrained weights already encode useful visual patterns. We keep those starting weights and train further with DETRAC examples. This is **transfer learning** or **fine-tuning**.

Pretrained does not mean no training happens. Training changes the weights. It is usually more practical than learning all visual patterns from random values.

## Step 4: Learn from mistakes

During a training batch:

1. Read a small group of training images and labels.
2. Predict vehicle boxes and types.
3. Calculate how far the predictions are from the reference answers.
4. Adjust weights to reduce those errors.
5. Repeat with the next group of images.

One **epoch** is one pass through all training examples. A **batch** is the group processed at one time. The **learning rate** controls the size of weight updates.

YOLO reports box loss (location error), classification loss (type error), and DFL, an additional term that helps learn precise box edges. A loss is a training error measure, not an accuracy percentage. A falling training loss alone does not prove the model generalizes.

## Step 5: Validate

After an epoch, the model predicts validation examples without updating its weights on them. Compare training losses and validation metrics. If training improves while validation worsens, the model may be memorizing training details; this is **overfitting**.

**Early stopping** stops a run when validation fitness has not improved for the configured patience. `best.pt` is the checkpoint selected by validation fitness; it is not necessarily the last epoch.

The two-epoch smoke run checks whether the code executes and produces a checkpoint. Its tiny sample is not enough to judge final quality. The full experiment restarts from pretrained initialization.

## Step 6: Test

Freeze your settings and selected checkpoint, then evaluate the test split.

- **Precision:** Of the detections the model made, how many were correct?
- **Recall:** Of the reference vehicles, how many did the model find?
- **IoU:** How much does a predicted box overlap a reference box? Intersection area divided by union area.
- **mAP50:** Average precision summarized over classes, matching boxes at IoU 0.50.
- **mAP50–95:** A stricter summary averaged over multiple overlap thresholds from 0.50 to 0.95.

Always report which data and ignore-region policy produced the result. Our masked-image evaluation differs from the official DETRAC protocol. Test performance on prepared masked images is not a guarantee for raw traffic photographs.

## Step 7: Use the model

A new image goes to the trained model. No XML labels are loaded. Keep detections above the display threshold, calculate centers, draw boxes, and export the CSV. Lower thresholds usually show more candidates and more false positives; higher thresholds can miss vehicles.

## Short explanation for class

“We start with a pretrained detector and train it further on labeled traffic images. We keep separate images for validation and final testing. After training, our program identifies vehicles in a new image, calculates their positions, and saves the annotated image and a CSV table.”
