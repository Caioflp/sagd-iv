## Running the experiments:

From this folder, run
```sh
  python -m pip install -r requirements.txt
  PYTHONPATH="${PYTHONPATH}:../" python src/scripts/benchmark_continuous_response.py
  PYTHONPATH="${PYTHONPATH}:../" python src/scripts/benchmark_binary_response.py
```
Results will be stored under `outputs/`.
