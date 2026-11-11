@echo off
REM Two experiments that matter for acceptance (run overnight, about 8-10 hours in total):
REM   Part 1  VeReMi NextGen: second dataset (other city, other attacks, geographic splits).
REM           Proposed model vs Bi-LSTM vs XGBoost, 3 seeds each, same pipeline, nothing tuned.
REM   Part 2  XGBoost tuned on val-dev (7 settings), best one re-tested with 3 seeds.
REM Safe to re-run after a power cut: every finished step is skipped.
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project

set NGZIP=
if exist F:\hiwa\nextgen_zips\InTAS_*.zip set NGZIP=F:\hiwa\nextgen_zips
if "%NGZIP%"=="" if exist F:\nextgen_zips\InTAS_*.zip set NGZIP=F:\nextgen_zips
if "%NGZIP%"=="" (
  echo NextGen zips not found in F:\hiwa\nextgen_zips or F:\nextgen_zips - Part 1 skipped.
  goto part2
)
set NG=F:\hiwa\nextgen
set NGC=F:\hiwa\nextgen_cache
set NGR=F:\hiwa\nextgen_results
echo ===== Part 1: NextGen from %NGZIP% =====
python step11_nextgen.py --nextgen %NGZIP% --work %NG% --stage all --workers 4
if errorlevel 1 (
  echo NextGen preparation stopped - send %NG%\nextgen_log.txt. Continuing with Part 2.
  goto part2
)
REM NextGen has 196.5 M messages: keep 25 %% of receivers (complete streams), same rows for every model
python step11c_nextgen_subsample.py --features %NG%\features --out %NG%\features_sub --fraction 0.25
if not exist %NGC%\info.json python step5a_prepare.py --features %NG%\features_sub --out %NGC%
if not exist %NGC%\gs_id.npy python step9a_scene_arrays.py --features %NG%\features_sub --cache %NGC%
python step9_train_mixed.py --cache %NGC% --out %NGR% --eval-split test --seeds 0,1,2 --lr 0.001 --name ngmix --mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s,m_road_log_density
python step5b_train_model.py --cache %NGC% --out %NGR% --eval-split test --seeds 0,1,2 --lr 0.001 --name ngbilstm --encoder lstm --neighbors 0
python step4_baselines.py --features %NG%\features_sub --out %NGR% --protocol vehicle --task multi --models xgb --seeds 0,1,2 --drop-features m_
python step11b_nextgen_summary.py --cache %NGC% --results %NGR% --work %NG%

:part2
echo ===== Part 2: XGBoost tuning on val-dev =====
python step12_xgb_tune.py --features F:\hiwa\veremi_features_v3 --out F:\hiwa\results_xgbtune --proposed F:\hiwa\results_mixed

echo.
echo Send these files:
echo   F:\hiwa\nextgen_results\nextgen_summary.txt
echo   F:\hiwa\nextgen\nextgen_log.txt
echo   F:\hiwa\results_xgbtune\xgb_tuning_summary.txt
pause
