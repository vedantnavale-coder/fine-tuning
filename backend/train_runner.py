from model_manager import ModelManager
import sys

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    base_model_id = sys.argv[2]
    output_name = sys.argv[3]

    manager = ModelManager()
    manager.train_model(dataset_path, base_model_id, output_name)