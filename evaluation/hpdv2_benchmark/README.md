# HPDv2 benchmark prompts

Prompts from [zhwang/HPDv2](https://huggingface.co/datasets/zhwang/HPDv2/tree/main/benchmark) (`benchmark/*.json`).

| File | Description |
|------|-------------|
| `prompts/drawbench.json` | DrawBench prompts |
| `prompts/photo.json` | Photo category prompts |
| `prompts/paintings.json` | Paintings category prompts |
| `prompts/concept-art.json` | Concept-art category prompts |
| `prompts/anime.json` | Anime category prompts |

Re-download prompts only:

```bash
export HF_ENDPOINT='https://hf-mirror.com'
huggingface-cli download zhwang/HPDv2 --repo-type dataset \
  --local-dir /mnt/afs_zhangyunzhe/TDM/evaluation/hpdv2_benchmark/_hf_cache \
  --include "benchmark/*.json"
cp evaluation/hpdv2_benchmark/_hf_cache/benchmark/*.json evaluation/hpdv2_benchmark/prompts/
```

Benchmark images (`benchmark_imgs/`, `drawbench/*.tar.gz`) are not included here.
