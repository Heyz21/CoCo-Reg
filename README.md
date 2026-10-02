# CoCo-Reg


## Project structure

```text
CoCo-Reg-Code/
├── training/
│   └── train_gpcreg.py
├── evaluation/
│   ├── evaluate_gpcreg.py
│   └── generate_distance_distributions.py
├── visualization/
│   ├── plot_distance_distributions.py
│   ├── plot_mean_distance_by_deformation.py
│   ├── visualize_registration_examples.py
│   └── visualize_failure_cases.py
├── models/
│   ├── DefTransNet.py
│   └── deformation.py
├── tests/
│   └── test_deformation.py
├── README.md
└── requirements.txt
```


## Data and checkpoints

Place the data and checkpoints under the project root as follows:

```text
CoCo-Reg-Code/
├── ModelNet10/
├── CoCo-Reg.pth
├── DeftransNet.pth
└── Robust_Trained.pth
```

Download ModelNet10: https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip

Download CheckPoints: https://drive.google.com/drive/folders/1cdBQaD9gjQ3Xx9hdArRps0yzz-6KutcY?usp=sharing

## Training

The default formal training uses seeds 42, 43 and 44 for 20 epochs:

```powershell
python -m training.train_gpcreg --data_root ModelNet10 --save_dir Registration --run_name gpcreg --seeds 42 43 44 --epochs 20 --device cuda:0
```

Results are saved under `Registration/gpcreg/seed_<seed>/`. 

## Formal evaluation

```powershell
python -m evaluation.evaluate_gpcreg --data_root ModelNet10 --robust_ckpt Robust_Trained.pth --deftransnet_ckpt DeftransNet.pth --gpc_ckpt CoCo-Reg.pth --gpc_script training/train_gpcreg.py --device cuda:0
```

The evaluator uses the fixed seed-42 split: 182 validation objects and 726
test objects. All models receive the same source, target and target order.

## Distribution data

```powershell
python -m evaluation.generate_distance_distributions --gpc_ckpt CoCo-Reg.pth --device cuda:0
```

## Figures

```powershell
python -m visualization.plot_mean_distance_by_deformation
python -m visualization.plot_distance_distributions
python -m visualization.visualize_registration_examples --gpc_ckpt CoCo-Reg.pth --device cuda:0
python -m visualization.visualize_failure_cases --gpc_ckpt CoCo-Reg.pth --device cuda:0
```

## Deformation check

```powershell
python -m tests.test_deformation --sample ModelNet10/desk/test/desk_0202.off --progression
```

## Notes

- `Robust_Trained.pth` is the official pretrained checkpoint released by the authors.
- `DeftransNet.pth` is the 20th epoch DefTransNet checkpoint trained with the original notebook `DeftransNet.ipynb`.
- `CoCo-Reg.pth` is the  20th epoch CoCo-Reg checkpoint.

