@echo off
REM ===================================================================
REM  Step 13 - reviewer analyses (no training; saved models only).
REM  Part A: unseen ConstPosOffset / GridSybil / DelayedMessages.
REM  Part B: Sybil variants on MixAll. About 30-60 minutes.
REM  Put step13_reviewer_analysis.py in F:\hiwa\vanet_project first.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
python step13_reviewer_analysis.py --cache F:\hiwa\dl_cache --mixed F:\hiwa\results_mixed ^
    --main-features F:\hiwa\veremi_features_v3 --mixall-features F:\hiwa\mixall\features ^
    --out F:\hiwa\results_mixed
echo.
echo Done. Send this file: F:\hiwa\results_mixed\reviewer_analysis.txt
pause
