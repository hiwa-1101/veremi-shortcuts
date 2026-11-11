@echo off
REM XGBoost (vehicle split) with the 26 features + ONE extra field at a time.
REM Shows which non-behavioural field gives the 93.86% of the literature protocol.
REM 3 runs, about 35 min each. Finished runs are skipped on re-run.
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
set TB=--features F:\veremi_features_v3 --out F:\results_extra --eval-split test --task multi --models xgb --seeds 0 --protocol vehicle
python step4_baselines.py %TB% --extra-cols aux_pos_x,aux_pos_y --tag _litpos
python step4_baselines.py %TB% --extra-cols senderPseudo --tag _litpseudo
python step4_baselines.py %TB% --extra-cols rcvTime --tag _littime
python step6_summary.py --extra F:\results_extra
echo.
echo Send the section "4. Literature protocol" from F:\results_extra\extra_summary.txt
pause
