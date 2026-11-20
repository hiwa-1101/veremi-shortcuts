@echo off
REM ===================================================================
REM  Second context architecture (Bi-LSTM + GraphSAGE neighbourhood).
REM  Does mixed-scene training also remove the scene shortcut here?
REM  9 trainings (3 regimes x 3 seeds) + MixAll test + summary.
REM  Restartable: finished runs are skipped. Estimated 3-6 hours.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
set B=F:\hiwa\GAN
set OUT=%B%\results_sage

REM ---- check that all inputs exist before starting ----
set MISSING=
for %%P in ("%B%\dl_cache\info.json" "%B%\dl_cache\gs_id.npy" "%B%\results_final" "%B%\results_mixed" "%B%\results_extra" "%B%\mixall\features" "%B%\veremi_features_v3") do (
  if not exist %%P set MISSING=1& echo NOT FOUND: %%P
)
if defined MISSING (
  echo.
  echo Some inputs are missing. Send me a screenshot of this window and of: dir %B% /b
  pause
  exit /b 1
)

if not exist %OUT% mkdir %OUT%
set COMMON=--cache %B%\dl_cache --out %OUT% --eval-split test --seeds 0,1,2 --lr 0.001 --encoder lstm --nb-agg sage
set FAILED=

echo [1/5] original training, 26 inputs
python step5b_train_model.py %COMMON% --name lsage || set FAILED=%FAILED% 1
echo [2/5] mixed-scene training, 26 inputs
python step9_train_mixed.py %COMMON% --name lsage_mix --mix-q 0.5 || set FAILED=%FAILED% 2
echo [3/5] mixed-scene training, density inputs removed
python step9_train_mixed.py %COMMON% --name lsage_mixnd --mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s || set FAILED=%FAILED% 3
echo [4/5] MixAll test with vehicle-level labels (new models + references)
python step8c_mixall_variants.py --features %B%\mixall\features --main-features %B%\veremi_features_v3 --cache %B%\dl_cache --final %B%\results_final --extra %B%\results_extra --mixed %B%\results_mixed --out %OUT% --variants "lsage:%OUT%:0,1,2;lsage_mix:%OUT%:0,1,2;lsage_mixnd:%OUT%:0,1,2;bilstm:extra:0,1,2;tmodel:final:0,1,2,3,4;tmix:mixed:0,1,2;tmix_nodens:mixed:0,1,2" || set FAILED=%FAILED% 4
echo [5/5] summary of the main test split
python step15_sage_summary.py --sage %OUT% --final %B%\results_final --mixed %B%\results_mixed --extra %B%\results_extra || set FAILED=%FAILED% 5

echo.
if defined FAILED (echo FAILED steps:%FAILED% - send me a screenshot) else (echo All steps finished.)
echo Send me: %OUT%\sage_summary.txt and %OUT%\mixall_variants.txt
pause
