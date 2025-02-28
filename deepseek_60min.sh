model_name=TimeLLM
train_epochs=30
learning_rate=0.01
llama_layers=16

master_port=01188
num_process=4
batch_size=2
d_model=8
d_ff=32

comment='TimeLLM-Glucose'


accelerate launch  --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --config_file ./accelerate_config_4gpus.yaml run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 591_bbm_train.csv \
  --test_data_path 591_bbm_test.csv \
  --model_id 591_BBM_60MINS_Deepseek \
  --model TimeLLM \
  --model_comment "$comment" \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 18 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 7 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --llm_model DEEPSEEK \
  --llm_dim 4096 \
  --use_amp