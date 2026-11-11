@echo off
REM ===================================================================
REM  Full data pipeline + development runs (run AFTER the download ends)
REM  Usage: double-click, or in CMD:  F:\vanet_project\run_pipeline.bat
REM  Every step skips work that is already done, so it is safe to re-run.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project

echo.
echo ===== [1/6] Extract all archives =====
python step2a_extract_all.py --zips F:\veremi_zips --out F:\veremi || goto :error

echo.
echo ===== [2/6] Build raw labeled tables =====
python step2b_build_raw.py --root F:\veremi --out F:\veremi_parquet --workers 6 || goto :error

echo.
echo ===== [3/6] Features + leakage-free splits (v3) =====
python step3_features_splits.py --raw F:\veremi_parquet --out F:\veremi_features_v3 --workers 6 || goto :error

echo.
echo ===== [4/6] Prepare arrays for the deep model =====
if not exist F:\dl_cache\info.json (
    python step5a_prepare.py --features F:\veremi_features_v3 --out F:\dl_cache || goto :error
) else (
    echo dl_cache already exists - skipped
)

echo.
echo ===== [5/6] XGBoost baseline on validation (all classes) =====
python step4_baselines.py --features F:\veremi_features_v3 --out F:\results_dev_v3 --models xgb --seeds 0 --eval-split val || goto :error

echo.
echo ===== [6/6] Proposed model on validation =====
python step5b_train_model.py --cache F:\dl_cache --out F:\results_dev_v3 --eval-split val --seeds 0 || goto :error

echo.
echo ===== ALL DONE =====
echo Send these files from F:\results_dev_v3 :
echo    xgb_multi_vehicle_val_seed0.json
echo    tmodel_multi_vehicle_val_seed0.json
pause
exit /b 0

:error
echo.
echo ***** A step failed. Copy the error text above and send it. *****
pause
exit /b 1
