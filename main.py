import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

from src.gee.gee_handler import init_gee
from src.models.lstm_model import run_lstm_pipeline
from src.models.cnn_model import run_cnn_pipeline
from src.models.inference import draw_heatmap_and_confusion

def main():
    print("Pipeline started.")

    init_gee()

    print("Running LSTM model...")
    run_lstm_pipeline()

    print("Running CNN model...")
    heatmap, y_true, y_pred, y_scores = run_cnn_pipeline()

    print("Generating evaluation plots...")
    draw_heatmap_and_confusion(heatmap, y_true, y_pred, y_scores)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
