@echo off
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
python step7_inspect.py --features F:\veremi_features_v3 --mixall F:\mixall_zip --nextgen F:\nextgen_zips --out F:\results_extra
echo.
echo Send the file F:\results_extra\inspect_report.txt
pause
