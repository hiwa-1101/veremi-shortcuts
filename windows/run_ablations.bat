@echo off
REM ===================================================================
REM  Model + ablations on validation (val-dev), seed 0, learning rate 1e-3
REM  (the earlier lr=2e-3 made training unstable: loss rose after epoch 1)
REM  5 runs, about 15 min each. Finished runs are skipped if re-run.
REM ===================================================================
call F:\vanet_env\Scripts\activate
cd /d F:\vanet_project
set OUT=F:\results_dev_lr1e3
set COMMON=--cache F:\dl_cache --out %OUT% --eval-split val --seeds 0 --lr 0.001

echo ===== [1/5] Full model =====
python step5b_train_model.py %COMMON% || goto :error
echo ===== [2/5] No link history (L=1) =====
python step5b_train_model.py %COMMON% --history 1 --tag _noHist || goto :error
echo ===== [3/5] No neighbour branch (M=0) =====
python step5b_train_model.py %COMMON% --neighbors 0 --tag _noNeigh || goto :error
echo ===== [4/5] Wider neighbourhood (M=64, W=2 s) =====
python step5b_train_model.py %COMMON% --neighbors 64 --window 2.0 --tag _M64W2 || goto :error
echo ===== [5/5] Longer link history (L=32) =====
python step5b_train_model.py %COMMON% --history 32 --tag _L32 || goto :error

echo.
echo ===== ALL DONE =====
echo Send all files named tmodel*_val_seed0.json from %OUT%
pause
exit /b 0

:error
echo ***** A step failed. Copy the error text above and send it. *****
pause
exit /b 1
