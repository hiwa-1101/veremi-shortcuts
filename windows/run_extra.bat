@echo off
REM ===================================================================
REM  ADDITIONAL experiments requested by reviewers (test split).
REM  Uses the frozen configuration: L=16, M=32, W=1 s, lr=0.001.
REM  Results go to F:\results_extra. Finished runs are skipped on re-run.
REM  A failed step is reported but does NOT stop the remaining steps.
REM  Total roughly 8-9 hours: run overnight.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
set OUT=F:\results_extra
set DL=--cache F:\dl_cache --out %OUT% --eval-split test --lr 0.001
set TB=--features F:\veremi_features_v3 --out %OUT% --eval-split test --task multi --models xgb --seeds 0
set FAILED=

echo ===== [1/7] McNemar tests + vehicle bootstrap (about 5 min) =====
if not exist %OUT%\stats_mcnemar_bootstrap.json python step6_stats.py --cache F:\dl_cache --results F:\results_final --out %OUT% || set FAILED=%FAILED% 1

echo ===== [2/7] int8 quantisation (about 20 min) =====
if not exist %OUT%\quant_int8_seed0.json python step6_quantize.py --cache F:\dl_cache --results F:\results_final --out %OUT% --seed 0 || set FAILED=%FAILED% 2

echo ===== [3/7] Permutation importance / shortcut check (about 30 min) =====
if not exist %OUT%\importance_seed0.json python step6_importance.py --cache F:\dl_cache --results F:\results_final --out %OUT% --seed 0 || set FAILED=%FAILED% 3

echo ===== [4/7] Bi-LSTM literature baseline, 3 seeds (about 1 h) =====
python step5b_train_model.py %DL% --seeds 0,1,2 --name bilstm --encoder lstm --neighbors 0 || set FAILED=%FAILED% 4

echo ===== [5/7] Traffic-density transfer (about 1.5 h) =====
for %%A in (0709 1416) do (
  for %%B in (0709 1416) do (
    python step5b_train_model.py %DL% --seeds 0 --train-period %%A --test-period %%B --tag _d%%Ato%%B || set FAILED=%FAILED% 5
    python step4_baselines.py %TB% --protocol vehicle --train-period %%A --test-period %%B --tag _d%%Ato%%B || set FAILED=%FAILED% 5
  )
)

echo ===== [6/7] Literature protocol: XGBoost + position, pseudonym ID, time (about 1.2 h) =====
python step4_baselines.py %TB% --protocol random  --extra-cols aux_pos_x,aux_pos_y,senderPseudo,rcvTime --tag _lit || set FAILED=%FAILED% 6
python step4_baselines.py %TB% --protocol vehicle --extra-cols aux_pos_x,aux_pos_y,senderPseudo,rcvTime --tag _lit || set FAILED=%FAILED% 6

echo ===== [7/7] Leave-one-attack-out, proposed model, 19 attacks (about 4 h) =====
for /L %%K in (1,1,19) do (
  echo ----- unseen attack %%K -----
  python step5b_train_model.py %DL% --seeds 0 --exclude-attack %%K --tag _loao%%K || set FAILED=%FAILED% 7
)

echo ===== Summary =====
python step6_summary.py --extra %OUT%

echo.
if defined FAILED (
  echo ***** These steps had an error:%FAILED%  - copy the error text above and send it. *****
) else (
  echo ===== ALL DONE =====
)
echo Send the file F:\results_extra\extra_summary.txt
echo and a zip of all .json files in F:\results_extra
pause
