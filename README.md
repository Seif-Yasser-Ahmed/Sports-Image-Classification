# Technology-Sales-Data



## Download Dataset
### Windows
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\download_dataset.ps1 -Slug sidharkal/sports-image-classification
```

### Linux
```bash
./download_dataset.sh sidharkal/sports-image-classification
```

## Tensorboard
in the logs dir
```
tensorboard --logdir=logs
```
