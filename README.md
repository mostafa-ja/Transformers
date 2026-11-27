# Transformers — a study series

Six notebooks from April 2023, studying the transformer architecture the way it deserves to be studied: **build one by hand, take a pretrained one apart, use it off the shelf, fine-tune it, change how it was pretrained, and finally change how the pretraining itself is organised.**

They were written while I was learning this material, following the sources credited in each notebook, with my own notes, detours and checks kept in place. Outputs are preserved from the original runs. For the deeper from-scratch work that came later, see [LLM_from_scratch](https://github.com/mostafa-ja/LLM_from_scratch).

## Reading order

| # | Notebook | What it does |
|---|---|---|
| 01 | [`01_transformer_from_scratch.ipynb`](01_transformer_from_scratch.ipynb) | The full transformer implemented by hand in PyTorch: embeddings, positional encoding, multi-head attention, encoder, decoder, assembly, inference. |
| 02 | [`02_vision_transformer_walkthrough.ipynb`](02_vision_transformer_walkthrough.ipynb) | A pretrained ViT taken apart with `timm`: patches, position embeddings (visualised), attention maps, the classification head. |
| 03 | [`03_vit_inference_huggingface.ipynb`](03_vit_inference_huggingface.ipynb) | The same ViT through Hugging Face in a dozen cells: load, preprocess, forward pass, prediction. |
| 04 | [`04_fine_tune_vit_beans.ipynb`](04_fine_tune_vit_beans.ipynb) | Fine-tuning ViT on the beans dataset with `Trainer` — plus a weight-by-weight comparison of what fine-tuning actually changed. |
| 05 | [`05_fine_tune_vit_mae_retina.ipynb`](05_fine_tune_vit_mae_retina.ipynb) | The same recipe with a masked-autoencoder (ViTMAE) backbone on retinal disease images. |
| 06 | [`06_ssl_federated_learning.ipynb`](06_ssl_federated_learning.ipynb) | Fed-MAE: MAE pretraining done federated (SSL-FL), then fine-tuning, on retinal data. |

The arc in one line: **01–02 are how a transformer works, 03–04 are how you use and adapt one, 05–06 are where its representations come from.**

## Sources

Each notebook credits its source in its introduction: Arun Mohan's Kaggle transformer-from-scratch tutorial (01), Hiroto Honda's ViT walkthrough (02), the Hugging Face documentation and the "Fine-Tune ViT" blog post (03–05), and the SSL-FL repository by Rui Yan and co-authors (06). The explanations, experiments and the fine-tuning weight comparison in 04 are my own.

## A note on versions

These notebooks date from April 2023 and use the APIs of that time — for example `ViTFeatureExtractor` and `datasets.load_metric` in the fine-tuning notebooks, whose modern equivalents are `AutoImageProcessor` and `evaluate.load`. The code is kept exactly as run; treat the notebooks as a dated study record, not as current-API reference.
