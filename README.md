# GlucoseLLM

The model leverages large language models (LLMs) like LLaMA, GPT-2, BERT, and DeepSeek to enhance its forecasting capabilities. This project focuses on short-term and long-term glucose level forecasting, which is crucial for diabetes management.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Miniconda**: A package and environment management system.

### Installation

1. **Clone the Repository**  

   ```bash 
   git clone https://github.com/your-username/TimeLLM.git
   cd GlucoseLLM
   ```

2. **Create a 'checkpoints' Directory**  

   Ensure you have a `checkpoints/` directory where the trained models will be saved.

3. **Create a Conda Environment**  

   ```bash
   conda create -n glucose -c conda-forge -c nvidia python=3.11 cuda-toolkit=12.3
   ```

4. **Activate Conda Environment**  

   ```bash
   conda activate glucose
   ```

5. **Install Dependencies**  

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset used in this project is located in the `./dataset/glucose/` directory. Ensure you have the following files:

- `544_bbm_train.csv`
- `544_bbm_test.csv`
- `540_bbm_train.csv`
- `540_bbm_test.csv`

## Configuration

### Forecasting Horizons  

- **30-minute horizon:**  
  ```bash
  --label_len 12  
  --pred_len 6  
  ```  

- **60-minute horizon:**  
  ```bash
  --label_len 18  
  --pred_len 12  
  ```  

## Training Command

To train the **TimeLLM** model using multiple GPUs with mixed precision (`bf16`), run the following command:  

```bash
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 4 --main_process_port 1188 --config_file ./accelerate_config_4gpus.yaml run_glucose.py   --task_name short_term_forecast   --is_training 1   --root_path ./dataset/glucose/   --data_path 544_bbm_train.csv   --test_data_path 544_bbm_test.csv   --model_id 544_BBM_BS2   --model TimeLLM   --data Glucose   --features S   --separate_test no   --seq_len 432   --label_len 12   --pred_len 6   --e_layers 2   --d_layers 1   --factor 3   --enc_in 1   --dec_in 7   --c_out 1   --d_model 8   --d_ff 32   --batch_size 2   --learning_rate 0.01   --llm_layers 16   --train_epochs 30   --model_comment "TimeLLM-Glucose"
```

## Testing Command

To run the model in testing mode, change `--is_training` to `0` in the command:

```bash
--is_training 0
```

## Running the Model

To start training or testing, use the following command:

```bash
bash {filename}.sh
```

For example:

```bash
bash timellm_glucose2020_bbm_test_v2.sh
```

## Notes

- Adjust `--label_len` and `--pred_len` for different forecasting horizons.  
- Modify `--batch_size` and `--learning_rate` based on hardware availability and convergence behavior.  
