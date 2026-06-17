"""Decisive test: does passing explicit position_ids to llm.generate() break SP long-gen?
bs=1 (left-pad path is a no-op here), greedy, with vs without position_ids."""
import torch, re
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

dev = "cuda"
m = OpenTSLMSP(llm_id="meta-llama/Llama-3.2-1B", device=dev)
m.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
m = m.to(dev)
ck = torch.load("results/Llama_3_2_1B/OpenTSLMSP/stage3_cot/checkpoints/best_model.pt",
                map_location="cpu", weights_only=False)
m.encoder.load_state_dict(ck["encoder_state"]); m.projector.load_state_dict(ck["projector_state"])
m.load_lora_state_from_checkpoint(ck, allow_missing=True); m.eval()

def gen(item, use_pos):
    batch = extend_time_series_to_match_patch_size_and_aggregate([item], patch_size=m.patch_size)
    ie, am = m.pad_and_apply_batch(batch)
    gk = dict(inputs_embeds=ie, attention_mask=am, max_new_tokens=512)
    if use_pos:
        gk["position_ids"] = (am.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        ids = m.llm.generate(**gk)
    return m.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]

ds = HARCoTQADataset(split="test", EOS_TOKEN=m.tokenizer.eos_token or "<|end_of_text|>")
for i in range(3):
    item = ds[i]
    print(f"\n######## SAMPLE {i}")
    for use_pos in (True, False):
        o = gen(item, use_pos)
        a = re.search(r"Answer:\s*([A-Za-z_ ]+)", o)
        print(f"  [position_ids={use_pos!s:5s}] len={len(o):5d} Answer={(a.group(1).strip() if a else None)!r}")
        print(f"     tail={o[-80:]!r}")
