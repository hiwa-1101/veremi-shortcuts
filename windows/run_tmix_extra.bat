@echo off
REM ===================================================================
REM  Repeats the key analyses for the FINAL model (mixed-scene training),
REM  so every result in the paper refers to the same model.
REM  About 5.5 hours: run overnight. Finished runs are skipped on re-run.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
set OUT=F:\hiwa\results_mixed
set MIX=--cache F:\hiwa\dl_cache --out %OUT% --eval-split test --lr 0.001 --name tmix --mix-q 0.5
set FAILED=

echo ===== [1/4] McNemar + vehicle bootstrap vs XGBoost (5 min) =====
if not exist %OUT%\stats_mcnemar_bootstrap_tmix.json python step6_stats.py --cache F:\hiwa\dl_cache --results F:\hiwa\results_final --results-a %OUT% --model tmix --out %OUT% || set FAILED=%FAILED% 1

echo ===== [2/4] Permutation importance (30 min) =====
if not exist %OUT%\importance_tmix_seed0.json python step6_importance.py --cache F:\hiwa\dl_cache --results %OUT% --model tmix --out %OUT% --seed 0 || set FAILED=%FAILED% 2

echo ===== [3/4] Traffic-period transfer (30 min) =====
for %%A in (0709 1416) do (
  for %%B in (0709 1416) do (
    python step9_train_mixed.py %MIX% --seeds 0 --train-period %%A --test-period %%B --tag _d%%Ato%%B || set FAILED=%FAILED% 3
  )
)

echo ===== [4/4] Leave-one-attack-out, 19 attacks (about 4 h) =====
for /L %%K in (1,1,19) do (
  echo ----- unseen attack %%K -----
  python step9_train_mixed.py %MIX% --seeds 0 --exclude-attack %%K --tag _loao%%K || set FAILED=%FAILED% 4
)

python step6_summary.py --extra %OUT%
echo.
if defined FAILED (echo ***** Steps with an error:%FAILED% *****) else (echo ===== ALL DONE =====)
echo Send F:\hiwa\results_mixed\extra_summary.txt
pause
