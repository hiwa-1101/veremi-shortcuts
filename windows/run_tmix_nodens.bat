@echo off
REM Control run: final model WITHOUT the two traffic-density inputs (n_msgs_1s, n_pseudos_1s),
REM then all variants on MixAll and main test. About 55 minutes.
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
python step9b_density_shared.py F:\hiwa\results_mixed F:\hiwa\results_extra > F:\hiwa\results_mixed\density_shared.txt
type F:\hiwa\results_mixed\density_shared.txt
python step9_train_mixed.py --cache F:\hiwa\dl_cache --out F:\hiwa\results_mixed --eval-split test --seeds 0,1,2 --lr 0.001 --name tmix_nodens --mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s
python step8c_mixall_variants.py --features F:\hiwa\mixall\features --main-features F:\hiwa\veremi_features_v3 --cache F:\hiwa\dl_cache --final F:\hiwa\results_final --extra F:\hiwa\results_extra --mixed F:\hiwa\results_mixed --out F:\hiwa\results_mixed
echo.
echo Send: F:\hiwa\results_mixed\density_shared.txt and F:\hiwa\results_mixed\mixall_variants.txt
pause
