@echo off
REM Reviewer checks on the FINAL training regime (mixed-scene, 24 inputs):
REM   (a) neighbourhood ablation M=0, 3 seeds   (b) q = 0.25 and 0.75, seed 0
REM Then MixAll + main test for all variants, and a summary. About 1.5-2 hours.
REM Safe to re-run after a power cut: finished runs are skipped.
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
set OUT=F:\hiwa\results_mixed
set ZF=--zero-features n_msgs_1s,n_pseudos_1s

for %%S in (0 1 2) do (
  if not exist %OUT%\tmix_nodens_noNeigh_multi_vehicle_seed%%S_weights.pt (
    python step9_train_mixed.py --cache F:\hiwa\dl_cache --out %OUT% --eval-split test --seeds %%S --lr 0.001 --name tmix_nodens_noNeigh --neighbors 0 --mix-q 0.5 %ZF%
  )
)
if not exist %OUT%\tmix_nodens_q25_multi_vehicle_seed0_weights.pt (
  python step9_train_mixed.py --cache F:\hiwa\dl_cache --out %OUT% --eval-split test --seeds 0 --lr 0.001 --name tmix_nodens_q25 --mix-q 0.25 %ZF%
)
if not exist %OUT%\tmix_nodens_q75_multi_vehicle_seed0_weights.pt (
  python step9_train_mixed.py --cache F:\hiwa\dl_cache --out %OUT% --eval-split test --seeds 0 --lr 0.001 --name tmix_nodens_q75 --mix-q 0.75 %ZF%
)

python step8c_mixall_variants.py --features F:\hiwa\mixall\features --main-features F:\hiwa\veremi_features_v3 --cache F:\hiwa\dl_cache --final F:\hiwa\results_final --extra F:\hiwa\results_extra --mixed %OUT% --out %OUT%
python step10_ablation_summary.py %OUT%
echo.
echo Send these two files:  %OUT%\ablation_q_summary.txt   and   %OUT%\mixall_variants.txt
pause
