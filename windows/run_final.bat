@echo off
REM ===================================================================
REM  FINAL experiments for the paper - TEST split (used only here, once)
REM  Configuration frozen on validation: L=16, M=32, W=1 s, lr=0.001
REM  Ordered by importance; total roughly 10-12 hours (run overnight).
REM  Safe to re-run: finished runs are skipped.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
set OUT=F:\results_final
set DL=--cache F:\dl_cache --out %OUT% --eval-split test --lr 0.001
set TB=--features F:\veremi_features_v3 --out %OUT% --eval-split test

echo ===== [1/6] Proposed model, 5 seeds (about 1 h) =====
python step5b_train_model.py %DL% --seeds 0,1,2,3,4 || goto :error

echo ===== [2/6] XGBoost, vehicle-disjoint split, 3 seeds (about 2 h) =====
python step4_baselines.py %TB% --protocol vehicle --task multi --models xgb --seeds 0,1,2 || goto :error

echo ===== [3/6] XGBoost, RANDOM split = leakage measurement, 3 seeds (about 2 h) =====
python step4_baselines.py %TB% --protocol random --task multi --models xgb --seeds 0,1,2 || goto :error

echo ===== [4/6] Ablations on test, 3 seeds (about 1 h) =====
python step5b_train_model.py %DL% --seeds 0,1,2 --history 1 --tag _noHist || goto :error
python step5b_train_model.py %DL% --seeds 0,1,2 --neighbors 0 --tag _noNeigh || goto :error

echo ===== [5/6] LightGBM, 3 seeds =====
python step4_baselines.py %TB% --protocol vehicle --task multi --models lgbm --seeds 0,1,2 || goto :error

echo ===== [6/6] Random Forest, 3 seeds =====
python step4_baselines.py %TB% --protocol vehicle --task multi --models rf --seeds 0,1,2 || goto :error

echo ===== Figures and tables from the final results =====
python make_figures.py --results %OUT% --eval-split test --counts step3_class_split_counts.csv --audit step2c_audit.json --out F:\paper_figures_final

echo.
echo ===== ALL DONE =====
echo Zip the folders F:\results_final (only the .json files) and F:\paper_figures_final and send them.
pause
exit /b 0

:error
echo ***** A step failed. Copy the error text above and send it. *****
pause
exit /b 1
