"""Diagnostic: is SP's long-CoT generation collapse a DECODING problem (fixable with
repetition_penalty/no_repeat) or a model/training problem? Loads SP stage3 exactly as
curriculum eval does (encoder+projector+LoRA), generates a few HAR test items bs=1 under
several decoding configs. A correct HAR CoT reaches 'Answer: <activity>' in ~250 tokens."""
import os, re, torch
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

dev = "cuda"
m = OpenTSLMSP(llm_id="meta-llama/Llama-3.2-1B", device=dev)
m.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
m = m.to(dev)
ck = torch.load("results/Llama_3_2_1B/OpenTSLMSP/stage3_cot/checkpoints/best_model.pt",
                map_location="cpu", weights_only=False)
m.encoder.load_state_dict(ck["encoder_state"])
m.projector.load_state_dict(ck["projector_state"])
nlora = m.load_lora_state_from_checkpoint(ck, allow_missing=True)
m.eval()
print(f"loaded lora params={nlora}  ckpt val_loss={ck.get('val_loss')}  epoch={ck.get('epoch')}")
print(f"eos_token_id={m.tokenizer.eos_token_id}  eos={m.tokenizer.eos_token!r}")

ds = HARCoTQADataset(split="test", EOS_TOKEN=m.tokenizer.eos_token or "<|end_of_text|>")

def gen(item, **kw):
    batch = extend_time_series_to_match_patch_size_and_aggregate([item], patch_size=m.patch_size)
    with torch.no_grad():
        return m.generate(batch, max_new_tokens=512, **kw)[0]

CONFIGS = [
    ("greedy",      {}),
    ("rep_pen=1.3", dict(repetition_penalty=1.3)),
    ("norepeat=3",  dict(no_repeat_ngram_size=3)),
    ("sample t=0.7",dict(do_sample=True, temperature=0.7, top_p=0.9)),
]
for i in range(3):
    item = ds[i]
    gold = re.search(r"Answer:\s*([A-Za-z_ ]+)", item["answer"])
    print(f"\n######## SAMPLE {i}  gold={gold.group(1).strip() if gold else '?'!r}")
    for tag, kw in CONFIGS:
        o = gen(item, **kw)
        a = re.search(r"Answer:\s*([A-Za-z_ ]+)", o)
        print(f"  [{tag:12s}] len={len(o):5d}  Answer={(a.group(1).strip() if a else None)!r}")
        print(f"               head={o[:90]!r}")
        print(f"               tail={o[-70:]!r}")
