"""Demo script to showcase CheXNet capabilities."""

import torch
from chexnet.model import DenseNet121, ResNet18, build_model
from chexnet.engine import train, test
from chexnet.visualize import visualize_prediction
from chexnet.dataset import CLASSES, CLASS_TO_IDX, PNEUMONIA_IDX
import argparse
import os


def main():
    parser = argparse.ArgumentParser("CheXNet Demo")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ChestX-ray14 directory")
    parser.add_argument("--train_csv", type=str, default="train.csv",
                        help="Training CSV path")
    parser.add_argument("--val_csv", type=str, default="val.csv",
                        help="Validation CSV path")
    parser.add_argument("--test_csv", type=str, default="test.csv",
                        help="Test CSV path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for Grad-CAM visualization")
    parser.add_argument("--arch", type=str, choices=["DENSE-NET-121", "RES-NET-18"],
                        default="DENSE-NET-121")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (demo)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_tencrop", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_model(args.arch, class_count=14, pretrained=True).to(device)
    print(f"Model: {args.arch}")

    # Quick test evaluation
    if args.test_csv and os.path.exists(args.test_csv):
        print("\n=== Test Evaluation ===")
        test(args.data_dir, args.test_csv, path_model=args.checkpoint or "best_model.pth",
             arch=args.arch, use_tencrop=args.use_tencrop)

    # Grad-CAM visualization
    if args.image and os.path.exists(args.image):
        print(f"\n=== Grad-CAM Visualization for: {args.image} ===")
        try:
            fig, top_classes = visualize_prediction(
                model, args.image, class_names=CLASSES, top_k=3, device=device)
            plt.show()
            print("Top predicted diseases:")
            for cls, prob in top_classes:
                print(f"  {cls}: {prob:.4f}")
        except Exception as e:
            print(f"Visualization error: {e}")

    # Print architecture summary
    print(f"\n=== Model Architecture: {args.arch} ===")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()