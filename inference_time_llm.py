#!/usr/bin/env python3
import argparse
import os

import torch
import numpy as np
import pandas as pd

from accelerate import Accelerator
from peft import PeftModel

from data_provider.data_factory import data_provider
from models.TimeLLM import Model as TimeLLMModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Time‑LLM Inference with LoRA Adapter + CSV Export"
    )

    # ── Data / inference settings ──────────────────────────────────────────────
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset key for data_provider (e.g. Weather)')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--data_path', type=str, required=True,
                        help='CSV filename for inference (e.g. weather_inf.csv)')
    parser.add_argument('--seq_len',   type=int, default=96,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Prompt (label) length')
    parser.add_argument('--pred_len',  type=int, default=96,
                        help='Prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Inference batch size')
    parser.add_argument('--freq',      type=str, default='h',
                        help='Time feature frequency (s, t, h, d, …)')
    parser.add_argument('--device',    type=str, default=None,
                        help='Device (cuda or cpu)')

    # ── args required by data_provider ────────────────────────────────────────
    parser.add_argument('--embed',            type=str, default='timeF',
                        help='Time encoding (timeF, fixed, learned)')
    parser.add_argument('--features',         type=str, default='M',
                        help='Forecast type (M:multi→multi, S:single→single, MS:multi→single)')
    parser.add_argument('--target',           type=str, default='OT',
                        help='Target column (when features is S or MS)')
    parser.add_argument('--seasonal_patterns',type=str, default='Monthly',
                        help='For M4 only')
    parser.add_argument('--percent',          type=int, default=100,
                        help='Data usage percent')
    parser.add_argument('--num_workers',      type=int, default=0,
                        help='DataLoader workers')

    # ── TimeLLM architecture settings ────────────────────────────────────────
    parser.add_argument('--task_name',   type=str, default='long_term_forecast',
                        help='Task name (matches training)')
    parser.add_argument('--model',       type=str, default='TimeLLM',
                        help='Model name (TimeLLM)')
    parser.add_argument('--enc_in',      type=int, required=True,
                        help='Encoder input size (num of features)')
    parser.add_argument('--d_model',     type=int, required=True,
                        help='Model embedding dimension (d_model)')
    parser.add_argument('--n_heads',     type=int, required=True,
                        help='Number of attention heads')
    parser.add_argument('--d_ff',        type=int, required=True,
                        help='Feed‑forward dimension (d_ff)')
    parser.add_argument('--patch_len',   type=int, default=16,
                        help='Patch length')
    parser.add_argument('--stride',      type=int, default=8,
                        help='Patch stride')
    parser.add_argument('--dropout',     type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--prompt_domain',type=int, default=0,
                        help='Use custom prompt domain? (0 or 1)')

    # ── LLM & LoRA settings ─────────────────────────────────────────────────
    parser.add_argument('--llm_model',   type=str, required=True,
                        help='LLM backbone (LLAMA, GPT2, BERT, QWEN)')
    parser.add_argument('--llm_layers',  type=int, required=True,
                        help='Number of layers in the LLM')
    parser.add_argument('--llm_dim',     type=int, required=True,
                        help='LLM hidden dim')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='LoRA adapter dir (contains adapter_config.json + .safetensors)')

    return parser.parse_args()

def main():
    args = parse_args()

    accelerator = Accelerator()
    device = torch.device(
        args.device if args.device else accelerator.device
    )

    # prepare data loader
    args.is_training = 0
    args.batch_size = args.batch_size
    _, test_loader = data_provider(args, 'test')

    # build model
    model = TimeLLMModel(args).to(device)
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    

    # load adapter
    model.llm_model = PeftModel.from_pretrained(
        model.llm_model,
        args.checkpoint,
    )
    model.llm_model.eval()

    # inference
    all_preds = []
    with torch.no_grad():
        iterator = test_loader
        if accelerator.is_local_main_process:
            from tqdm import tqdm
            iterator = tqdm(
                test_loader,
                desc="🔍 Inference",
                unit="batch",
                dynamic_ncols=True
            )
        for bx, by, bxm, bym in iterator:
            bx  = bx.float().to(device)
            by  = by.float().to(device)
            bxm = bxm.float().to(device)
            bym = bym.float().to(device)

            dec_inp = torch.zeros_like(by[:, -args.pred_len:, :]).to(device)
            dec_inp = torch.cat([by[:, :args.label_len, :], dec_inp], dim=1)

            out = model(bx, bxm, dec_inp, bym)
            all_preds.append(out.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)  # [N, pred_len, C]

    # save CSV
    csv_path = os.path.join(args.root_path, args.data_path)
    df_orig = pd.read_csv(csv_path)
    feature_cols = df_orig.columns[-preds.shape[-1]:].tolist()

    n_s, p_len, n_f = preds.shape
    flat = preds.reshape(-1, n_f)

    df_pred = pd.DataFrame(flat, columns=feature_cols)
    df_pred.insert(0, 'step',   list(range(p_len)) * n_s)
    df_pred.insert(0, 'sample', sum([[i]*p_len for i in range(n_s)], []))

    out_csv = 'time_llm_lora_preds.csv'
    df_pred.to_csv(out_csv, index=False)
    print(f"[Done] Saved predictions to {out_csv}")

if __name__ == "__main__":
    main()
