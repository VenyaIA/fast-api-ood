from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from PIL import Image

matplotlib.use('Agg')

model = YOLO('model/obb_model_s_yolo.pt')


def detect_objects(image_file):
    image = Image.open(image_file)
    results = model(image)
    annotated_img = results[0].plot()  # Use .plot() instead of .render()
    angle_degrees = []

    for result in results:
        if result.obb is not None and len(result.obb) != 0:
            for obb in result.obb:
                xywhr = obb.xywhr[0]
                x_center, y_center, width, height, rotation = xywhr
                angle_degree = rotation.item() * (180 / np.pi)
                angle_degrees.append(f"{angle_degree:.2f}")

                print(f"Angle: {angle_degree:.2f} degrees")
        else:
            print("No oriented bounding boxes detected.")

    plt.axis('off')

    if angle_degrees:
        return angle_degrees, annotated_img
    else:
        return None, annotated_img

