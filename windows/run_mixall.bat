@echo off
REM External test on MixAll_0024 with the final model (no retraining).
REM About 1-3 hours. Needs about 40 GB free on F: while running. Safe to re-run.
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
python step8_mixall.py --zip F:\mixall_zip\MixAll_0024.zip --work F:\mixall --cache F:\dl_cache --results F:\results_final --roadmap F:\vanet_project\road_map.parquet --out F:\results_extra
echo.
echo Send the file F:\results_extra\mixall_report.txt
pause
