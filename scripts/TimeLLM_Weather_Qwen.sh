model_name=TimeLLM
llm_model=QWEN
train_epochs=1
learning_rate=0.01
Qwen_layers=48
llm_dim=5120

master_port=10097
num_process=4
batch_size=10
d_model=16
d_ff=32

comment='TimeLLM-Weather'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather2.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --llm_dim $llm_dim \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $Qwen_layers \
  --llm_model $llm_model \
  --lora \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --train_epochs $train_epochs \
  --model_comment $comment