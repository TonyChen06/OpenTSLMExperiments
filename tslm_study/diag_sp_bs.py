"""Confirm the fix is complete for BATCHED eval: left-pad realign (as generate() does) but
do NOT pass position_ids. All samples in a mixed-length bs>1 batch should reach 'Answer: X'
and match their bs=1 outputs."""
import torch, re
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

dev = "cuda"
m = OpenTSLMSP(llm_id="meta-llama/Llama-3.2-1B", device=dev)
m.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0); m = m.to(dev)
ck = torch.load("results/Llama_3_2_1B/OpenTSLMSP/stage3_cot/checkpoints/best_model.pt",
                map_location="cpu", weights_only=False)
m.encoder.load_state_dict(ck["encoder_state"]); m.projector.load_state_dict(ck["projector_state"])
m.load_lora_state_from_checkpoint(ck, allow_missing=True); m.eval()

def batched_gen(items, pass_pos):
    batch = extend_time_series_to_match_patch_size_and_aggregate(items, patch_size=m.patch_size)
    ie, am = m.pad_and_apply_batch(batch)
    B, Lp, _ = ie.shape
    lengths = am.long().sum(1)
    if (lengths != Lp).any():                      # left-pad realign (copied from generate())
        ie2 = torch.zeros_like(ie); am2 = torch.zeros_like(am)
        for i in range(B):
            n = int(lengths[i]); ie2[i, Lp-n:] = ie[i, :n]; am2[i, Lp-n:] = am[i, :n]
        ie, am = ie2, am2
    gk = dict(inputs_embeds=ie, attention_mask=am, max_new_tokens=512)
    if pass_pos:
        gk["position_ids"] = (am.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        ids = m.llm.generate(**gk)
    return m.tokenizer.batch_decode(ids, skip_special_tokens=True)

ds = HARCoTQADataset(split="test", EOS_TOKEN=m.tokenizer.eos_token or "<|end_of_text|>")
items = [ds[i] for i in range(4)]   # mixed prompt lengths
print("=== BATCH of 4, NO position_ids (proposed fix) ===")
for i, o in enumerate(batched_gen(items, pass_pos=False)):
    a = re.search(r"Answer:\s*([A-Za-z_ ]+)", o)
    print(f"  [{i}] len={len(o):4d} Answer={(a.group(1).strip() if a else None)!r}")
print("=== same BATCH, WITH position_ids (current, for contrast) ===")
for i, o in enumerate(batched_gen(items, pass_pos=True)):
    a = re.search(r"Answer:\s*([A-Za-z_ ]+)", o)
    print(f"  [{i}] len={len(o):4d} Answer={(a.group(1).strip() if a else None)!r}")
