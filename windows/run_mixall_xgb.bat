@echo off
REM XGBoost baseline on MixAll (trained on the main training set exactly as before,
REM then applied unchanged to MixAll with vehicle-level labels). About 40 minutes.
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
python step4_baselines.py --features F:\veremi_features_v3 --out F:\results_extra --eval-split test --task multi --models xgb --seeds 0 --protocol vehicle --tag _mixall --external-features F:\mixall\features > F:\results_extra\mixall_xgb_log.txt 2>&1
powershell -command "Select-String -Path F:\results_extra\mixall_xgb_log.txt -Pattern 'EXTERNAL|MAIN test|Error|Traceback'"
echo.
echo Send the lines above (or the file F:\results_extra\mixall_xgb_log.txt)
pause
