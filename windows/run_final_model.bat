@echo off
REM ===================================================================
REM  All analyses for the FINAL model: mixed-scene training without the two
REM  density inputs (tmix_nodens). About 6 hours: run overnight.
REM  Finished runs are skipped on re-run (also after a power cut).
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\hiwa\vanet_project
set OUT=F:\hiwa\results_mixed
set FIN=--cache F:\hiwa\dl_cache --out %OUT% --eval-split test --lr 0.001 --name tmix_nodens --mix-q 0.5 --zero-features n_msgs_1s,n_pseudos_1s
set FAILED=

echo ===== [1/5] McNemar + vehicle bootstrap vs XGBoost (5 min) =====
if not exist %OUT%\stats_mcnemar_bootstrap_tmix_nodens.json python step6_stats.py --cache F:\hiwa\dl_cache --results F:\hiwa\results_final --results-a %OUT% --model tmix_nodens --out %OUT% || set FAILED=%FAILED% 1

echo ===== [2/5] int8 quantisation (20 min) =====
if not exist %OUT%\quant_int8_tmix_nodens_seed0.json python step6_quantize.py --cache F:\hiwa\dl_cache --results %OUT% --model tmix_nodens --out %OUT% --seed 0 || set FAILED=%FAILED% 2

echo ===== [3/5] Permutation importance (30 min) =====
if not exist %OUT%\importance_tmix_nodens_seed0.json python step6_importance.py --cache F:\hiwa\dl_cache --results %OUT% --model tmix_nodens --out %OUT% --seed 0 || set FAILED=%FAILED% 3

echo ===== [4/5] Traffic-period transfer (30 min) =====
for %%A in (0709 1416) do (
  for %%B in (0709 1416) do (
    python step9_train_mixed.py %FIN% --seeds 0 --train-period %%A --test-period %%B --tag _d%%Ato%%B || set FAILED=%FAILED% 4
  )
)

echo ===== [5/5] Leave-one-attack-out, 19 attacks (about 4.3 h) =====
for /L %%K in (1,1,19) do (
  echo ----- unseen attack %%K -----
  python step9_train_mixed.py %FIN% --seeds 0 --exclude-attack %%K --tag _loao%%K || set FAILED=%FAILED% 5
)

python step9b_density_shared.py %OUT% F:\hiwa\results_extra > %OUT%\density_shared.txt
python step6_summary.py --extra %OUT%
type %OUT%\density_shared.txt >> %OUT%\extra_summary.txt
echo.
if defined FAILED (echo ***** Steps with an error:%FAILED% *****) else (echo ===== ALL DONE =====)
echo Send F:\hiwa\results_mixed\extra_summary.txt
pause
