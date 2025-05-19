import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

import torch
from src.gee.gee_handler import init_gee
#from src.models.lstm_model import run_lstm_pipeline
from src.models.cnn_model import run_cnn_pipeline_multi, train_model
from src.models.inference import draw_heatmap_and_confusion

def main(train=False):
    print("Pipeline started.")

    # Initialize Earth Engine
    init_gee()

    # Run LSTM pipeline
 #   print("Running LSTM model...")
#    run_lstm_pipeline()

    # CNN Eğitim (isteğe bağlı)
    #if train:
   #     print("Training CNN model...")
     #   train_model(
    #        train_path="/content/drive/MyDrive/food_crisis_prediction2/data/processed/train",
   #         val_path="/content/drive/MyDrive/food_crisis_prediction2/data/processed/val",
   #         model_save_path="/content/drive/MyDrive/food_crisis_prediction2/models/cnn_resnet18_drought3.pth",
      #      epochs=15,
     #       batch_size=64,
    #        learning_rate=1e-4,
   #         patience=3
  #      )


    # Run CNN inference
    print("Running CNN model...")
    tif_folder = "/content/drive/MyDrive/GEE_NDVI"
    results = run_cnn_pipeline_multi(tif_folder)

    # Evaluation plots
    print("Generating evaluation plots...")
    for i, result in enumerate(results):
        if not result["y_scores"]:  # Skip empty results
            print(f"Skipping {result['file']} (no valid predictions)")
            continue

        draw_heatmap_and_confusion(
            heatmap=result["heatmap"],
            y_true=result["y_true"],
            y_pred=result["y_pred"],
            y_scores=result["y_scores"],
            title_suffix=f"_{result['file']}"
        )

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    # Eğitim yapılacaksa train=True
    main(train=True)
