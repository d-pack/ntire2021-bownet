# NTIRE 2021 Track 1 Debluring

This code is based on [ESGAN-FS](https://github.com/ManuelFritsche/real-world-sr).
For more information visit their repository.


### Train
To train model:
* change paths to data in *esrgan-fs/codes/options/ntire2021/pretrain.yml*
* run train script:
```bash
$ cd esrgan-fs/codes
$ python train.py -opt options/ntire2021/pretrain.yml
```
Result will be in folder *esrgan-fs/experiments/BOWGAN_pretrain*

### Test

Pretrained model can be found [here](https://drive.google.com/file/d/1-LeHuMvxM7TxIGp0PwL2mrEH_w9i9C02)

To test model:
* change paths to data and the model in *esrgan-fs/codes/options/ntire2021/test_for_submit.yml*
* run test script:
```bash
$ cd esrgan-fs/codes
$ python m_test.py -opt options/ntire2021/test_for_submit.yml
```
Result will be in folder *esrgan-fs/results/BOWGAN_pretrain*

### Results on NTIRE 2021  Track 1 Debluring

Our results:  

| PSNR | SSIM | LPIPS | TIME |  
|---|---|---|---|  
| 27.054675 | 0.852373 | 0.343938 | 0.214931 |

Result data can be found [here](https://drive.google.com/file/d/1iMRj7iWsUjSosMJp9qjMFv1Ay5SkoFlA)
