# Gemma3 Packing Training Benchmark

This run compares ordinary fixed-length Tunix SFT batches against packed batches using Default CE only.

- Dataset: `opus100-en-fr-gemma3-it`
- Source: Helsinki-NLP/opus-100 en-fr train split, Tunix Gemma3 IT prompt wrapper, target-only loss mask, target EOS
- Model: `google/gemma-3-12b-it`
- Tokenizer source: `sentencepiece`

![training_comparison](training_comparison.png)

## Summary

| Variant | Steps | Batch | Max length | Fit examples | Rows/batches | Final loss | Eval loss | BLEU | chrF | Step time | Valid tok/s | Loss tok/s | Packing density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unpacked | 50 | 1 | 512 | 4999 | 4999 | 1.6663 | nan | nan | nan | 0.134s | 277 | 95 | 10.5% |
