# HPSv2 offline weights

Weights from [xswu/HPSv2](https://huggingface.co/xswu/HPSv2).

| File | Size | Notes |
|------|------|--------|
| `weights/HPS_v2.pt` | ~7.6 GB | Default full checkpoint (Space `app.py` default) |
| `weights/HPS_v2_compressed.pt` | ~1.9 GB | Compressed variant |
| `weights/HPS_v2.1_compressed.pt` | ~1.9 GB | HPS v2.1 compressed variant |

Re-download:

```bash
export HF_ENDPOINT='https://hf-mirror.com'
huggingface-cli download xswu/HPSv2 --local-dir /mnt/afs_zhangyunzhe/TDM/evaluation/hpsv2/_hf_cache
cp evaluation/hpsv2/_hf_cache/HPS_v2*.pt evaluation/hpsv2/weights/
```

## Offline load (example)

Requires `open_clip` / HPSv2 code (e.g. from [spaces/xswu/HPSv2](https://huggingface.co/spaces/xswu/HPSv2)):

```python
import torch
from open_clip import create_model_and_transforms, get_tokenizer

checkpoint_path = "/mnt/afs_zhangyunzhe/TDM/evaluation/hpsv2/weights/HPS_v2.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess_train, preprocess_val = create_model_and_transforms(
    "ViT-H-14",
    "laion2B-s32B-b79K",
    precision="amp",
    device=device,
    output_dict=True,
)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
tokenizer = get_tokenizer("ViT-H-14")
```

Use `preprocess_val` for images and tokenize prompts with `tokenizer` when scoring image–text pairs.
