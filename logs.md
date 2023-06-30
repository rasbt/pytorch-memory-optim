
# 01_pytorch-vit.py

- regular PyTorch script
- batch size 32

```
torch    : 2.1.0.dev20230623+cu118
lightning: 2.1.0.dev0

Torch CUDA available? True
Global seed set to 123
Files already downloaded and verified
Epoch: 0001/0001 | Batch 0000/0703 | Loss: 2.4105
Epoch: 0001/0001 | Batch 0300/0703 | Loss: 0.0640
Epoch: 0001/0001 | Batch 0600/0703 | Loss: 0.0680
Epoch: 0001/0001 | Train acc.: 94.44% | Val acc.: 96.00%
Time elapsed 17.94 min
Memory used: 26.79 GB
Test accuracy 95.85%
```

# 01-2_pytorch-fabric.py

- same as above but using fabric
- same results as expected

```
Epoch: 0001/0001 | Batch 0000/0703 | Loss: 2.4105
Epoch: 0001/0001 | Batch 0300/0703 | Loss: 0.1754
Epoch: 0001/0001 | Batch 0600/0703 | Loss: 0.2308
Epoch: 0001/0001 | Train acc.: 94.44% | Val acc.: 96.06%
Time elapsed 17.88 min
Memory used: 26.84 GB
Test accuracy 96.06%
```

# 02_mixed-precision.py

```
Epoch: 0001/0001 | Batch 0000/0703 | Loss: 2.4105
Epoch: 0001/0001 | Batch 0300/0703 | Loss: 0.1088
Epoch: 0001/0001 | Batch 0600/0703 | Loss: 0.1302
Epoch: 0001/0001 | Train acc.: 94.56% | Val acc.: 96.02%
Time elapsed 3.45 min
Memory used: 18.21 GB
Test accuracy 95.71%
```
# 03_bfloat16.py

```
Epoch: 0001/0001 | Batch 0000/0703 | Loss: 2.4101
Epoch: 0001/0001 | Batch 0300/0703 | Loss: 0.1149
Epoch: 0001/0001 | Batch 0600/0703 | Loss: 0.0269
Epoch: 0001/0001 | Train acc.: 95.62% | Val acc.: 96.94%
Time elapsed 2.88 min
Memory used: 13.82 GB
Test accuracy 96.92%
```

# 04_lower-batchsize.py

- batch size 16 (instead of 64 before)

```
Epoch: 0001/0001 | Batch 0000/2812 | Loss: 2.4567
Epoch: 0001/0001 | Batch 0300/2812 | Loss: 0.2379
Epoch: 0001/0001 | Batch 0600/2812 | Loss: 0.0248
Epoch: 0001/0001 | Batch 0900/2812 | Loss: 0.0716
Epoch: 0001/0001 | Batch 1200/2812 | Loss: 0.0398
Epoch: 0001/0001 | Batch 1500/2812 | Loss: 0.0177
Epoch: 0001/0001 | Batch 1800/2812 | Loss: 0.0273
Epoch: 0001/0001 | Batch 2100/2812 | Loss: 0.1532
Epoch: 0001/0001 | Batch 2400/2812 | Loss: 0.0085
Epoch: 0001/0001 | Batch 2700/2812 | Loss: 0.0031
Epoch: 0001/0001 | Train acc.: 95.21% | Val acc.: 97.20%
Time elapsed 3.96 min
Memory used: 5.69 GB
Test accuracy 97.34%
```

# 05_gradient-accum.py

```
Epoch: 0001/0001 | Batch 0000/11250 | Loss: 0.6012
Epoch: 0001/0001 | Batch 0300/11250 | Loss: 0.0044
Epoch: 0001/0001 | Batch 0600/11250 | Loss: 0.0032
Epoch: 0001/0001 | Batch 0900/11250 | Loss: 0.0155
Epoch: 0001/0001 | Batch 1200/11250 | Loss: 0.0021
Epoch: 0001/0001 | Batch 1500/11250 | Loss: 0.0658
Epoch: 0001/0001 | Batch 1800/11250 | Loss: 0.0016
Epoch: 0001/0001 | Batch 2100/11250 | Loss: 0.0359
Epoch: 0001/0001 | Batch 2400/11250 | Loss: 0.0106
Epoch: 0001/0001 | Batch 2700/11250 | Loss: 0.0100
Epoch: 0001/0001 | Batch 3000/11250 | Loss: 0.2942
Epoch: 0001/0001 | Batch 3300/11250 | Loss: 0.0020
Epoch: 0001/0001 | Batch 3600/11250 | Loss: 0.0222
Epoch: 0001/0001 | Batch 3900/11250 | Loss: 0.0075
Epoch: 0001/0001 | Batch 4200/11250 | Loss: 0.1245
Epoch: 0001/0001 | Batch 4500/11250 | Loss: 0.0032
Epoch: 0001/0001 | Batch 4800/11250 | Loss: 0.0266
Epoch: 0001/0001 | Batch 5100/11250 | Loss: 0.0039
Epoch: 0001/0001 | Batch 5400/11250 | Loss: 0.0014
Epoch: 0001/0001 | Batch 5700/11250 | Loss: 0.0171
Epoch: 0001/0001 | Batch 6000/11250 | Loss: 0.0009
Epoch: 0001/0001 | Batch 6300/11250 | Loss: 0.0021
Epoch: 0001/0001 | Batch 6600/11250 | Loss: 0.0655
Epoch: 0001/0001 | Batch 6900/11250 | Loss: 0.0004
Epoch: 0001/0001 | Batch 7200/11250 | Loss: 0.0003
Epoch: 0001/0001 | Batch 7500/11250 | Loss: 0.0004
Epoch: 0001/0001 | Batch 7800/11250 | Loss: 0.0011
Epoch: 0001/0001 | Batch 8100/11250 | Loss: 0.0106
Epoch: 0001/0001 | Batch 8400/11250 | Loss: 0.0321
Epoch: 0001/0001 | Batch 8700/11250 | Loss: 0.0018
Epoch: 0001/0001 | Batch 9000/11250 | Loss: 0.0004
Epoch: 0001/0001 | Batch 9300/11250 | Loss: 0.0013
Epoch: 0001/0001 | Batch 9600/11250 | Loss: 0.0001
Epoch: 0001/0001 | Batch 9900/11250 | Loss: 0.0003
Epoch: 0001/0001 | Batch 10200/11250 | Loss: 0.1277
Epoch: 0001/0001 | Batch 10500/11250 | Loss: 0.0005
Epoch: 0001/0001 | Batch 10800/11250 | Loss: 0.0007
Epoch: 0001/0001 | Batch 11100/11250 | Loss: 0.0490
Epoch: 0001/0001 | Train acc.: 95.46% | Val acc.: 97.24%
Time elapsed 12.91 min
Memory used: 3.91 GB
Test accuracy 97.27%
```

# 06 scheduler

# 07 init module

07-01_init_module.py
Without Fabric
CPU Memory used: 0.60 GB
GPU Memory used: 1.24 GB

07-02_init_module.py
Without init_module
CPU Memory used: 1.15 GB
GPU Memory used: 0.65 GB

07-03_init_module.py
With init_module
CPU Memory used: 0.70 GB
GPU Memory used: 0.65 GB



# 08 FSDP ADAM

- compares to 01-02


```
Epoch: 0001/0001 | Batch 0000/0175 | Loss: 2.4957
Epoch: 0001/0001 | Batch 0050/0175 | Loss: 0.1717
Epoch: 0001/0001 | Batch 0100/0175 | Loss: 0.0793
Epoch: 0001/0001 | Batch 0150/0175 | Loss: 0.1426
Epoch: 0001/0001 | Train acc.: 94.74% | Val acc.: 97.28%
Time elapsed 5.53 min
Memory used: 6.59 GB
Test accuracy 97.13%
```




# 09 FSDP Offload

- compares to 01-02

```
Epoch: 0001/0001 | Batch 0000/0175 | Loss: 2.4957
Epoch: 0001/0001 | Batch 0050/0175 | Loss: 0.1717
Epoch: 0001/0001 | Batch 0100/0175 | Loss: 0.0794
Epoch: 0001/0001 | Batch 0150/0175 | Loss: 0.1454
Epoch: 0001/0001 | Train acc.: 94.75% | Val acc.: 97.24%
Time elapsed 8.34 min
Memory used: 6.03 GB
Test accuracy 97.23%
```

## 10

```
Epoch: 0001/0001 | Batch 0000/0175 | Loss: 2.4957
Epoch: 0001/0001 | Batch 0050/0175 | Loss: 0.1717
Epoch: 0001/0001 | Batch 0100/0175 | Loss: 0.0803
Epoch: 0001/0001 | Batch 0150/0175 | Loss: 0.1496
Epoch: 0001/0001 | Train acc.: 94.74% | Val acc.: 97.38%
Time elapsed 8.48 min
Memory used: 6.03 GB
Test accuracy 97.22%
```

# bonus: distilbert before

Epoch: 0001/0001 | Batch 0000/0729 | Loss: 0.7085
Epoch: 0001/0001 | Batch 0300/0729 | Loss: 0.4883
Epoch: 0001/0001 | Batch 0600/0729 | Loss: 0.3101
Epoch: 0001/0001 | Train acc.: 90.55% | Val acc.: 92.99%
Test accuracy: 92.95%
Total training time: 0.71 min
Memory used: 3.99 GB

# bonus distilbert after

Epoch: 0001/0001 | Batch 0000/2916 | Loss: 0.1748
Epoch: 0001/0001 | Batch 0300/2916 | Loss: 0.1943
Epoch: 0001/0001 | Batch 0600/2916 | Loss: 0.0161
Epoch: 0001/0001 | Batch 0900/2916 | Loss: 0.0294
Epoch: 0001/0001 | Batch 1200/2916 | Loss: 0.0195
Epoch: 0001/0001 | Batch 1500/2916 | Loss: 0.1030
Epoch: 0001/0001 | Batch 1800/2916 | Loss: 0.0184
Epoch: 0001/0001 | Batch 2100/2916 | Loss: 0.0093
Epoch: 0001/0001 | Batch 2400/2916 | Loss: 0.0197
Epoch: 0001/0001 | Batch 2700/2916 | Loss: 0.0417
Epoch: 0001/0001 | Train acc.: 85.37% | Val acc.: 91.43%
Test accuracy: 90.43%
Total training time: 5.55 min
Memory used: 1.15 GB

# bonus bigbird before

N/A

# bonus bigbird after

Epoch: 0001/0001 | Batch 0000/8750 | Loss: 0.0697
Epoch: 0001/0001 | Batch 0300/8750 | Loss: 0.0469
Epoch: 0001/0001 | Batch 0600/8750 | Loss: 0.0557
Epoch: 0001/0001 | Batch 0900/8750 | Loss: 0.0181
Epoch: 0001/0001 | Batch 1200/8750 | Loss: 0.0254
Epoch: 0001/0001 | Batch 1500/8750 | Loss: 0.0125
Epoch: 0001/0001 | Batch 1800/8750 | Loss: 0.0138
Epoch: 0001/0001 | Batch 2100/8750 | Loss: 0.0050
Epoch: 0001/0001 | Batch 2400/8750 | Loss: 0.0120
Epoch: 0001/0001 | Batch 2700/8750 | Loss: 0.0048
Epoch: 0001/0001 | Batch 3000/8750 | Loss: 0.0114
Epoch: 0001/0001 | Batch 3300/8750 | Loss: 0.0221
Epoch: 0001/0001 | Batch 3600/8750 | Loss: 0.0027
Epoch: 0001/0001 | Batch 3900/8750 | Loss: 0.0035
Epoch: 0001/0001 | Batch 4200/8750 | Loss: 0.0015
Epoch: 0001/0001 | Batch 4500/8750 | Loss: 0.0014
Epoch: 0001/0001 | Batch 4800/8750 | Loss: 0.0082
Epoch: 0001/0001 | Batch 5100/8750 | Loss: 0.0041
Epoch: 0001/0001 | Batch 5400/8750 | Loss: 0.0023
Epoch: 0001/0001 | Batch 5700/8750 | Loss: 0.0059
Epoch: 0001/0001 | Batch 6000/8750 | Loss: 0.0036
Epoch: 0001/0001 | Batch 6300/8750 | Loss: 0.0023
Epoch: 0001/0001 | Batch 6600/8750 | Loss: 0.0011
Epoch: 0001/0001 | Batch 6900/8750 | Loss: 0.1048
Epoch: 0001/0001 | Batch 7200/8750 | Loss: 0.0050
Epoch: 0001/0001 | Batch 7500/8750 | Loss: 0.0027
Epoch: 0001/0001 | Batch 7800/8750 | Loss: 0.0042
Epoch: 0001/0001 | Batch 8100/8750 | Loss: 0.0016
Epoch: 0001/0001 | Batch 8400/8750 | Loss: 0.0022
Epoch: 0001/0001 | Batch 8700/8750 | Loss: 0.0016
Epoch: 0001/0001 | Train acc.: 88.69% | Val acc.: 93.28%
Test accuracy: 93.10%
Total training time: 75.94 min
Memory used: 4.03 GB