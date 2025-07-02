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
   --data_path 540_bbm_train_v2.csv \
   --test_data_path 540_bbm_test_v2.csv \
   --model_id 540_BBM_BS2 \
   --model $model_name \
   --data Glucose \
   --features S \
   --separate_test no \
   --seq_len 432 \
   --label_len 12 \
   --pred_len 6 \
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
   --model_comment $comment

accelerate launch  --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --config_file ./accelerate_config_4gpus.yaml run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 544_bbm_train.csv \
  --test_data_path 544_bbm_test.csv \
  --model_id 544_BBM_BS2 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
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
  --model_comment $comment

accelerate launch  --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --config_file ./accelerate_config_4gpus.yaml run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 552_bbm_train_v2.csv \
  --test_data_path 552_bbm_test_v2.csv \
  --model_id 552_BBM_BS2 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
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
  --model_comment $comment

accelerate launch  --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --config_file ./accelerate_config_4gpus.yaml run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 567_bbm_train_v2.csv \
  --test_data_path 567_bbm_test_v2.csv \
  --model_id 567_BBM_BS2 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
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
  --model_comment $comment

accelerate launch  --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --config_file ./accelerate_config_4gpus.yaml run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 584_bbm_train_v2.csv \
  --test_data_path 584_bbm_test_v2.csv \
  --model_id 584_BBM_BS2 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
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
  --model_comment $comment


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port --config_file ./accelerate_config_4gpus.yaml run_glucose.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/glucose/ \
  --data_path 596_bbm_train_v2.csv \
  --test_data_path 596_bbm_test_v2.csv \
  --model_id 596_BBM_BS2 \
  --model $model_name \
  --data Glucose \
  --features S \
  --separate_test no \
  --seq_len 432 \
  --label_len 12 \
  --pred_len 6 \
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
  --model_comment $comment


