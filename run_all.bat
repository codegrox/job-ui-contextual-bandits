@echo off
set PRESET=%1
if "%PRESET%"=="" set PRESET=debug

python -m src.generate_arm_profiles --preset %PRESET% --seed 4014
python -m src.generate_companies    --preset %PRESET% --seed 4014
python -m src.generate_applicants   --preset %PRESET% --seed 4014
python -m src.generate_jobs         --preset %PRESET% --seed 4014
python -m src.generate_sessions     --preset %PRESET% --seed 4014

if "%PRESET%"=="debug" (
    for %%i in (0 1 2 3 4) do python -m src.simulate_rounds --preset %PRESET% --seed 4014 --chunk-id %%i
)

if "%PRESET%"=="medium" (
    for %%i in (0 1 2 3 4 5) do python -m src.simulate_rounds --preset %PRESET% --seed 4014 --chunk-id %%i
)

if "%PRESET%"=="full" (
    for %%i in (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14) do python -m src.simulate_rounds --preset %PRESET% --seed 4014 --chunk-id %%i
)

python -m src.merge_and_split --preset %PRESET% --seed 4014
python -m src.diagnostics --preset %PRESET% --seed 4014
python -m src.run_experiments --preset %PRESET% --mode online --seeds 4014
python -m src.evaluate_results --preset %PRESET% --mode online
python -m src.export_report_tables --preset %PRESET%
