# Gemma3 Packing Training Benchmark

This run compares ordinary fixed-length Tunix SFT batches against packed batches using Default CE only.

- Dataset: `opus100-en-fr-gemma3-it`
- Source: Helsinki-NLP/opus-100 en-fr train split, Tunix Gemma3 IT prompt wrapper, target-only loss mask, target EOS
- Model: `google/gemma-3-270m-it`
- Tokenizer source: `sentencepiece`

![training_comparison](training_comparison.png)

## Summary

| Variant | Steps | Batch | Max length | Fit examples | Rows/batches | Final loss | Eval loss | BLEU | chrF | Step time | Valid tok/s | Loss tok/s | Packing density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| packed | 1000 | 16 | 512 | 4999 | 33 | 0.0132 | 4.3301 | 14.19 | 40.21 | 0.108s | 75668 | 30966 | 99.3% |
