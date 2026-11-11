@echo off
REM ===================================================================
REM  Mixed-scene training (fix for the scene-level shortcut), then test
REM  on the main test set AND on MixAll. About 45-60 minutes. Safe to re-run.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
echo ===== [1/3] Scene arrays (once, about 5 min) =====
if not exist F:\hiwa\dl_cache\gs_id.npy python step9a_scene_arrays.py --features F:\hiwa\veremi_features_v3 --cache F:\hiwa\dl_cache
echo ===== [2/3] Mixed-scene training, 3 seeds (about 40 min) =====
python step9_train_mixed.py --cache F:\hiwa\dl_cache --out F:\hiwa\results_mixed --eval-split test --seeds 0,1,2 --lr 0.001 --name tmix --mix-q 0.5
echo ===== [3/3] All variants on MixAll and main test (about 10 min) =====
python step8c_mixall_variants.py --features F:\hiwa\mixall\features --main-features F:\hiwa\veremi_features_v3 --cache F:\hiwa\dl_cache --final F:\hiwa\results_final --extra F:\hiwa\results_extra --mixed F:\hiwa\results_mixed --out F:\hiwa\results_mixed
echo.
echo Send the file F:\hiwa\results_mixed\mixall_variants.txt
pause
