# Trajectory, Identity and Scene Shortcuts in V2X Misbehavior Detection

Code for the paper

> H. Hassanzadeh and E. Khezri, "Trajectory, Identity and Scene Shortcuts in V2X Misbehavior Detection:
> An Audit of VeReMi Extension and a Mixed-Scene Link–Neighbourhood Transformer," submitted, 2026.

The repository contains everything needed to reproduce the paper:

- the data audit of VeReMi Extension;
- the leakage-free, vehicle-disjoint protocol;
- the receiver-observable features;
- the baselines (Random Forest, LightGBM, XGBoost and a Bi-LSTM);
- the proposed link–neighbourhood Transformer with mixed-scene training;
- the external test on MixAll_0024 and the second dataset, VeReMi NextGen;
- every additional experiment, figure and table.

## Requirements

- Python 3.10 or newer and an NVIDIA GPU with at least 16 GB of memory. We used an Intel Core i9-13900KF with 64 GB
  RAM and an NVIDIA RTX 4080 (16 GB), on Windows 11 with Python 3.12 and PyTorch 2.11 (CUDA 12.8).
- The tree baselines run on CPU or GPU (`XGB_DEVICE=cpu|cuda`).
- Several hundred GB of free disk space for the extracted archives, intermediate tables and features.
- Install the Python packages with `pip install -r requirements.txt`.

## Data

| Dataset | Source | Used for |
| --- | --- | --- |
| VeReMi Extension (incl. MixAll_0024) | doi: 10.5281/zenodo.20090854 | main experiments; external mixed-attack test |
| VeReMi NextGen | https://veremi-dataset.github.io/veremi-nextgen | second dataset (other city, 15 attacks) |

Download the archives yourself; they are not redistributed here. `data/` holds the files that fix our protocol:

- `splits_vehicles.csv` maps each physical vehicle to train, validation or test (seed 42, 70/15/15). The same
  assignment is used in every attack folder.
- `step2c_audit.json` holds the audit results (findings A1–A10).
- `kept_receivers.csv` lists the 25% of NextGen receivers kept by `step11c` (seed 42).

If `splits_vehicles.csv` is present in the working folder, `step3` reuses it, so every run uses the identical split.

## Pipeline

All scripts are in `pipeline/` and are run from that folder. Every step skips work that is already finished, so
an interrupted run can simply be restarted. The `windows/` folder holds the exact batch files we ran. They use our
drive paths (`F:\...`), so change the paths before using them.

| Step | Script | What it does | Paper |
| --- | --- | --- | --- |
| 1 | `step2a_extract_all.py` | extract every VeReMi Extension archive | §3.1 |
| 2 | `step2b_build_raw.py` | one table per simulation: messages, receiver state, labels, oracle deviations (labels only) | §3.1 |
| 3 | `step2c_audit.py` | data audit (vehicle and message reuse, noise shortcut, …) | §3.2, Table 2 |
| 4 | `step3_features_splits.py` | vehicle-disjoint split, 26 receiver-observable features, training-only road map | §4.1–4.2 |
| 5 | `step4_baselines.py` | RF / LightGBM / XGBoost; vehicle or random split; literature fields; LOAO; period transfer | §6.1–6.3, §6.7, App. A |
| 6 | `step5a_prepare.py` | normalised arrays and orderings for the deep model | §5 |
| 7 | `step5b_train_model.py` | link–neighbourhood Transformer (original training), ablations, Bi-LSTM | §4.3, §6.4 |
| 8 | `step9a_scene_arrays.py`, `step9_train_mixed.py` | mixed-scene training (final model: `--mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s`) | §4.4 |
| 9 | `step6_stats.py` | McNemar test and vehicle-cluster bootstrap | §6.6 |
| 10 | `step6_importance.py`, `step6_quantize.py` | permutation importance; int8 quantisation | App. A.4, A.6 |
| 11 | `step7_inspect.py`, `step9b_density_shared.py` | timestamp / identity checks; shared-class period transfer | §6.7, App. A.2–A.3 |
| 12 | `step8_mixall.py`, `step8_diag.py`, `step8c_mixall_variants.py` | MixAll_0024 external test (vehicle-level labels, finding A11) | §6.5, Table 7 |
| 13 | `step10_ablation_summary.py` | neighbourhood ablation and q sensitivity under the final regime | §6.4, Table 6 |
| 14 | `step11_nextgen.py`, `step11c_nextgen_subsample.py`, `step11b_nextgen_summary.py` | VeReMi NextGen pipeline and results | §6.8, App. A.8 |
| 15 | `step12_xgb_tune.py` | XGBoost tuned on val-dev, then tested once | §5, §6.6 |
| 16 | `step13_reviewer_analysis.py` | unseen attacks: single-input AUC and open-set score on the representation; Sybil variants on MixAll | §6.5, §6.7, App. A.9 |
| 17 | `step6_summary.py`, `make_figures.py` | text summary of all experiments; publication figures and tables | all |

### Minimal example (main result)

```bash
cd pipeline
python step2a_extract_all.py --zips <veremi_zips> --out <veremi>
python step2b_build_raw.py --root <veremi> --out <veremi_parquet> --workers 6
python step3_features_splits.py --raw <veremi_parquet> --out <features> --workers 6
python step5a_prepare.py --features <features> --out <dl_cache>
python step9a_scene_arrays.py --features <features> --cache <dl_cache>
# proposed model (mixed-scene training, 24 effective inputs), 3 seeds, test split
python step9_train_mixed.py --cache <dl_cache> --out <results> --eval-split test --seeds 0,1,2 \
       --lr 0.001 --name tmix_nodens --mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s
# XGBoost baseline, same split and features
python step4_baselines.py --features <features> --out <results> --protocol vehicle --task multi --models xgb --seeds 0,1,2
```

### VeReMi NextGen

```bash
python step11_nextgen.py --nextgen <nextgen_zips> --work <ng> --stage all --workers 4
python step11c_nextgen_subsample.py --features <ng>/features --out <ng>/features_sub --fraction 0.25
python step5a_prepare.py --features <ng>/features_sub --out <ng_cache>
python step9a_scene_arrays.py --features <ng>/features_sub --cache <ng_cache>
python step9_train_mixed.py --cache <ng_cache> --out <ng_results> --eval-split test --seeds 0,1,2 --lr 0.001 \
       --name ngmix --mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s,m_road_log_density
python step5b_train_model.py --cache <ng_cache> --out <ng_results> --eval-split test --seeds 0,1,2 --lr 0.001 \
       --name ngbilstm --encoder lstm --neighbors 0
python step4_baselines.py --features <ng>/features_sub --out <ng_results> --protocol vehicle --task multi \
       --models xgb --seeds 0,1,2 --drop-features m_
python step11b_nextgen_summary.py --cache <ng_cache> --results <ng_results> --work <ng>
```

## Protocol rules built into the code

- **Split unit.** The split unit is the physical vehicle, with one assignment shared by all attack folders. The
  test split is used only with `--eval-split test`. Every design decision uses `val-dev`, the half of the
  validation vehicles that is not used for early stopping.
- **Excluded inputs.** Models never see absolute coordinates, pseudonym or vehicle identifiers, receive time,
  `*_noise` fields or the `oracle_` columns.
- **Mixed-scene training.** It swaps only senders from the training split and uses no labels. It is active only
  during training; in leave-one-attack-out runs the withheld attack is removed from the swap pool.
- **Seeds.** All seeds are fixed: 42 for the split, 12345 for the validation halves and 0–4 for model runs.

## Citation

```bibtex
@article{hassanzadeh2026shortcuts,
  author  = {Hassanzadeh, Hiwa and Khezri, Edris},
  title   = {Trajectory, Identity and Scene Shortcuts in {V2X} Misbehavior Detection: An Audit of {VeReMi Extension}
             and a Mixed-Scene Link--Neighbourhood Transformer},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT (see `LICENSE`). The datasets keep their own licences.
