# Local Colab Runtime

This project can use the Google Colab website as its notebook interface while
running Python, model downloads, training, and file writes on this laptop.

## Start

1. Run `start_colab_local_runtime.ps1` with PowerShell.
2. The connection URL is copied to the clipboard and written to
   `local_runtime_url.txt`.
3. Open `Week 3/notebooks/week3_baseline_and_general_controls.ipynb` on the Colab website.
4. Select **Connect**, then **Connect to local runtime**.
5. Paste the URL and connect.
6. Run the notebook from the first cell.

The local runtime uses the project directory as `THESIS_DIR`. Results are
written under `week3_balanced_lora_controls`, beside the notebook.

## Stop

Run `stop_colab_local_runtime.ps1`.

## Repair the environment

Run `setup_local_colab_environment.ps1` if the Python environment or ML
packages need to be reinstalled.

## Hardware note

This laptop has an NVIDIA GeForce GTX 1050 Ti with 4 GB VRAM. It can load the
Qwen 0.5B model in 4-bit mode, but training and full-dataset evaluation will be
substantially slower than a Colab T4 with 16 GB VRAM. Keep the laptop awake,
plugged in, and well ventilated during a run.

Only connect trusted notebooks to the local runtime. A connected notebook can
read and modify files accessible to your Windows account.
