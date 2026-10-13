#!/usr/bin/env python3
"""Post-result diagnostic of the first failed audit's probability calculation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError('Use a new diagnostic receipt.')
study = Path(__file__).resolve().parents[1]
results = study/'results'
metadata = json.loads((results/'run-metadata.json').read_text())
for name,digest in metadata['model_hashes'].items():
    assert hashlib.sha256((args.model_dir/name).read_bytes()).hexdigest() == digest
prompts = json.loads((results/'prompts.json').read_text())
batch = [p for p in prompts if len(p['input_ids']) == 9][:32]
with (results/'scores.csv').open(newline='') as file:
    saved = {int(row['prompt_id']):row for row in csv.DictReader(file)}
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
model = GPTNeoXForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
ids = torch.tensor([p['input_ids'] for p in batch])
with torch.inference_mode():
    plain = model.embed_out(model.gpt_neox(ids,use_cache=False,output_attentions=False).last_hidden_state[:,-1])
    captured = model.embed_out(model.gpt_neox(ids,use_cache=False,output_attentions=True).last_hidden_state[:,-1])
records = []
for i,p in enumerate(batch):
    x = captured[i]
    probability32 = x.softmax(-1).numpy()
    probability64 = x.to(torch.float64).softmax(-1).numpy()
    numpy_logits = x.numpy().astype(np.float64)
    raw = np.exp(numpy_logits-numpy_logits.max())
    numpy_probability = raw/raw.sum()
    error = np.abs(probability32.astype(np.float64)-numpy_probability)
    worst = int(error.argmax())
    top = int(x.argmax())
    records.append(dict(prompt_id=p['prompt_id'],full_vocabulary_logits_error=float((plain[i]-captured[i]).abs().max()),
        worst_token=worst,logit=float(x[worst]),maximum_logit=float(x.max()),
        probability_float32=float(probability32[worst]),probability_float64=float(probability64[worst]),
        probability_numpy_float64=float(numpy_probability[worst]),float32_float64_error=float(error[worst]),
        numpy_torch_float64_maximum_error=float(np.abs(numpy_probability-probability64).max()),
        float32_sum_in_float64=float(probability32.astype(np.float64).sum()),
        saved_top_probability=float(saved[p['prompt_id']]['intact_top_probability']),
        saved_top_error=abs(float(probability32[top])-float(saved[p['prompt_id']]['intact_top_probability']))))
receipt = dict(status='complete',post_result_diagnostic=True,scope='First 32 length-9 split inputs; intact only.',
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_forward_calls=2,rows=records,
    maximum_full_vocabulary_logits_error=max(r['full_vocabulary_logits_error'] for r in records),
    maximum_saved_top_error=max(r['saved_top_error'] for r in records),
    maximum_numpy_torch_float64_error=max(r['numpy_torch_float64_maximum_error'] for r in records),
    maximum_float32_float64_error=max(r['float32_float64_error'] for r in records))
args.output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k != 'rows'}))
