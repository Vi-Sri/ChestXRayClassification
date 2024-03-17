# Training 

If you are in a slurm system, you can use the sbatch.sh script to run the training experiments

For Training,
python3 train.py --config config.json

To resume training from a checkpoint
python3 train.py --config config.json -r /home/srinivi/crcv/saved/models/XRayNet/0316_210940/checkpoint-epoch17.pth

To select a specific GPU device
python3 train.py --config config.json -d 0

for Testing, Make the training false in config under data loader section and run the command, 
python3 test.py --config config.json -r /home/srinivi/crcv/saved/models/XRayNet/0316_210940/model_best.pth


Here is the config.json i wrote for the experiments, 
```
{
    "name": "XRayNet",
    "n_gpu": 1,

    "arch": {
        "type": "ResNet18Model",
        "args": {
            "num_classes": 2,
            "pretrained": false
        }
    },
    "data_loader": {
        "type": "XRayDataloader",
        "args":{
            "data_dir": "/home/sriniana/projects/mic/chest-pa1/DomainAdaptativeClassifier/xray_classification/dataset/chest_xray",
            "batch_size": 32,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 1,
            "training": true
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "cross_entropy",
    "metrics": [
        "accuracy"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 100,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": true
    }
}
```
