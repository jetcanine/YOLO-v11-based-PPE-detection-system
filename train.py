from ultralytics import YOLO
import os

def main():
    # 1. Initialize the YOLO model with a pre-trained weight. we are choosing latest YOLOv11 nano here
    model = YOLO("yolo11n.pt")
    
    # 2. absolute path to dataset configuration file.
    # Make sure that the YAML file (with train, val, test paths, and classes) is correct.
    data_yaml_path = os.path.abspath("datasets/dataset/data.yaml")
    
    # 3. Define the hyperparameter search space.
    # Each key is a hyperparameter name and its value is a tuple (min, max).
    # You can add more hyperparameters according to your needs.
    search_space = {
        "lr0": (1e-5, 1e-1),         # initial learning rate
        "lrf": (0.01, 1.0),           # final learning rate factor
        "momentum": (0.6, 0.98),      # momentum factor
        "weight_decay": (0.0, 0.001), # L2 regularization
        "warmup_epochs": (0.0, 5.0),    # warmup epochs
        "warmup_momentum": (0.0, 0.95), # warmup momentum
        "box": (0.02, 0.2),           # bounding box loss weight
        "cls": (0.2, 4.0),            # classification loss weight
        "dfl": (0.2, 4.0),            # distribution focal loss weight
        "degrees": (0.0, 45.0),       # rotation augmentation in degrees
        # You can also tune augmentations such as translation, scale, shear, hsv_h, hsv_s, hsv_v, etc.
        "translate": (0.0, 0.9),
        "scale": (0.0, 0.9),
        "shear": (0.0, 10.0),
        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.9),
        "fliplr": (0.0, 1.0),        # probability of horizontal flip
        "mosaic": (0.0, 1.0),         # mosaic augmentation probability
        "mixup": (0.0, 1.0)           # mixup augmentation probability
    }
    
    # 4. Run hyperparameter tuning.
    # The tune() method will perform several iterations (each a separate training run) to search for better hyperparameters.
    print("Starting hyperparameter tuning...")
    tune_results = model.tune(
        data=data_yaml_path,  # your dataset configuration file
        epochs=30,            # number of epochs for each tuning iteration
        iterations=300,       # number of tuning iterations (each uses a mutated hyperparameter set)
        optimizer="AdamW",    # you can change the optimizer if needed
        space=search_space,   # the defined search space
        plots=True,           # generate tuning plots
        save=True,            # save the tuning results and weights
        val=True              # run validation only on the final epoch for speed
    )
    
    print("Hyperparameter tuning completed!")
    print("Best hyperparameters found:")
    print(tune_results)
    
if __name__ == "__main__":
    main()
