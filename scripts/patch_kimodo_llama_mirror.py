#!/usr/bin/env python3
"""Patch Kimodo to use NousResearch/Meta-Llama-3-8B-Instruct (open mirror)
instead of gated meta-llama/Meta-Llama-3-8B-Instruct.

Run AFTER kimodo pip install and AFTER first download of MNTP adapter.
Safe to run multiple times (idempotent).
"""
import json, os, pathlib, sys

KIMODO_DIR = pathlib.Path(__import__("kimodo").__file__).parent

# 1. Patch llm2vec.py prepare_for_tokenization
llm2vec_py = KIMODO_DIR / "model" / "llm2vec" / "llm2vec.py"
if llm2vec_py.exists():
    text = llm2vec_py.read_text()
    old = 'if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":'
    new = 'if self.model.config._name_or_path in ["meta-llama/Meta-Llama-3-8B-Instruct", "NousResearch/Meta-Llama-3-8B-Instruct"]:'
    if old in text:
        llm2vec_py.write_text(text.replace(old, new))
        print("[patch] llm2vec.py: added NousResearch to prepare_for_tokenization")
    elif "NousResearch" in text:
        print("[patch] llm2vec.py: already patched")
    else:
        print("[patch] llm2vec.py: WARNING - pattern not found, skipping")

# 2. Patch adapter_config.json (downloaded at runtime, so just print instructions)
#    The actual adapter_config patch happens in engine code at load time.
print("[patch] adapter_config.json: will be patched at runtime by KimodoEngine")
