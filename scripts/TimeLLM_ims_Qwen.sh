model_name=TimeLLM
llm_model=QWEN
train_epochs=10
learning_rate=0.001
Qwen_layers=48
llm_dim=5120

master_port=10097
num_process=2
batch_size=12
d_model=32
d_ff=32

comment='TimeLLM-IMS'

enc_in=13
dec_in=13
c_out=13

seq_len=360
pred_len=60
label_len=$(($seq_len-$pred_len))

model_id="ims_${seq_len}_${pred_len}"

save_step=5000


accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ims/ \
  --data_path ims_logs.csv \
  --save_step $save_step \
  --model_id $model_id \
  --model $model_name \
  --data IMS \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --d_model $d_model \
  --d_ff $d_ff \
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
  --model_comment $comment \