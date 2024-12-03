from finetune_main import train_model, get_dataset_size
import os

if __name__ == "__main__":
    base_model_name = "gliner_multi-v2.1"
    base_model = f"urchade/{base_model_name}"
    custom_model_name = f"merged_v1_{base_model_name}"
    model_dir = f"models_generalist/{custom_model_name}"
    data_dir = f"generalist_data/{custom_model_name}_data"
    train_path = f"finetune_data/merged_v1.json"
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # calculate number of epochs
    num_steps = 50000
    batch_size = 8
    data_size = get_dataset_size(train_path)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    print(f"number of epochs : {num_epochs}")

    split_ratio = 0.9
    learning_rate = 5e-6
    weight_decay = 0.05
    compile_model = False

    train_model(base_model, 
                model_dir, 
                data_dir,
                train_path, 
                split_ratio, 
                learning_rate, 
                weight_decay, 
                batch_size, 
                num_epochs, 
                compile_model,
                dataloader_num_workers=8)