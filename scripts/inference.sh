#!/usr/bin/env bash
# inference.sh
# 실행 전에 chmod +x inference.sh

# ── Dataset & paths ─────────────────────────────────────────────────────────────
DATA_KEY=Weather
ROOT=./dataset/weather
CSV_IN=weather_inf.csv

# ── DataProvider 옵션 ───────────────────────────────────────────────────────────
EMBED=timeF
FEATURES=M
SEASON=Monthly
PERCENT=100
WORKERS=10

FREQ=h
SEQ=512
LABEL=48
PRED=96
BATCH=32

# ── TimeLLM 아키텍처 ────────────────────────────────────────────────────────────
ENC_IN=21     # weather_inf.csv의 feature 개수
D_MODEL=32
N_HEADS=8
D_FF=32
PATCH_LEN=16
STRIDE=8
DROPOUT=0.1
PROMPT_DOMAIN=0

# ── LLM & LoRA ─────────────────────────────────────────────────────────────────
LLM=QWEN
LAYERS=48
DIM=5120
ADAPTER_DIR=/home/user/2HG/Time-LLM/checkpoints/long_term_forecast_weather_512_96_TimeLLM_Weather_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/checkpoint_epoch2_iter9000

NUM_PROCESS=4
MASTER_PORT=10199

accelerate launch --multi_gpu --num_processes $NUM_PROCESS --main_process_port $MASTER_PORT inference_time_llm.py \
  --data               $DATA_KEY \
  --root_path          $ROOT \
  --data_path          $CSV_IN \
  --embed              $EMBED \
  --features           $FEATURES \
  --seasonal_patterns  $SEASON \
  --percent            $PERCENT \
  --num_workers        $WORKERS \
  --freq               $FREQ \
  --seq_len            $SEQ \
  --label_len          $LABEL \
  --pred_len           $PRED \
  --batch_size         $BATCH \
  --task_name          long_term_forecast \
  --model              TimeLLM \
  --enc_in             $ENC_IN \
  --d_model            $D_MODEL \
  --n_heads            $N_HEADS \
  --d_ff               $D_FF \
  --patch_len          $PATCH_LEN \
  --stride             $STRIDE \
  --dropout            $DROPOUT \
  --prompt_domain      $PROMPT_DOMAIN \
  --llm_model          $LLM \
  --llm_layers         $LAYERS \
  --llm_dim            $DIM \
  --checkpoint         $ADAPTER_DIR \
  --device             cuda
