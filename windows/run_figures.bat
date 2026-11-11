@echo off
REM Re-draws every figure and table for the FINAL model. Run AFTER run_final_model.bat.
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
python make_figures.py --results F:\hiwa\results_final F:\hiwa\results_extra F:\hiwa\results_mixed --eval-split test --mixall F:\hiwa\results_mixed\mixall_variants.json --out F:\hiwa\paper_figures_v2
powershell -command "Compress-Archive -Force -Path F:\hiwa\paper_figures_v2\* -DestinationPath F:\hiwa\paper_figures_v2.zip"
echo.
echo Send F:\hiwa\paper_figures_v2.zip
pause
