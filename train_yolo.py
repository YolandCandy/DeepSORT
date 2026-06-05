import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26 on UA-DETRAC dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to UA-DETRAC dataset YAML config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Base model to start training from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize YOLO26 model
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)
    
    # Train the model
    # Note: UA-DETRAC needs to be formatted in YOLO format (images and labels text files)
    # The --data argument should point to a .yaml file defining the path to the dataset
    print(f"Starting training on {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device="0",  # GPU 0, change if you have multiple or use CPU
        project="ua_detrac_training",
        name="yolo26_car_detection",
        exist_ok=True
    )
    
    print("Training complete!")
    print(f"Metrics: {results}")

if __name__ == "__main__":
    main()
