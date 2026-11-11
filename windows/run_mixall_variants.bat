@echo off
REM Evaluates all saved model variants on MixAll (no training). About 5-10 minutes.
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
python step8c_mixall_variants.py --features F:\mixall\features --main-features F:\veremi_features_v3 --cache F:\dl_cache --final F:\results_final --extra F:\results_extra --out F:\results_extra
echo.
echo Send the file F:\results_extra\mixall_variants.txt
pause
