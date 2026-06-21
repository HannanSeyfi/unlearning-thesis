# Test Runs

Small smoke tests for confirming that Colab can write back to GitHub.

## HelloWorld GitHub Write Test

1. In Google Colab, open **Secrets** and add a secret named `GITHUB_TOKEN`.
2. Use a GitHub token that has repository **Contents: Read and write** access
   for `HannanSeyfi/unlearning-thesis`.
3. Run these cells in Colab:

```python
!wget -q https://raw.githubusercontent.com/HannanSeyfi/unlearning-thesis/main/test-runs/github_hello_world.py -O github_hello_world.py
!python github_hello_world.py
```

If it works, GitHub will show:

```text
test-runs/HelloWorld.txt
```

To test without writing to GitHub:

```python
!python github_hello_world.py --dry-run
```
