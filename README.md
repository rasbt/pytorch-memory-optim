# Optimizing PyTorch Memory Usage


This code repository contains the code used for my "Optimizing Memory Usage for Training LLMs and Vision Transformers in PyTorch" blog post. 

You can install the dependencies via


```bash
pip install -r requirements.txt
```

The scripts are all standalone scripts and can be run by executing 

```bash
python 01_pytorch-vit.py
```

and so forth. The only requirement is to have the [`local_utilities.py`](local_utilities.py) in the same folder as the script as it contains some data loading utilities that are reused across all the scripts.

I tracked the script outputs in the [`logs.md`](logs.md) file.
