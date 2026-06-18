# Publish to GitHub

This folder is the GitHub-ready copy of the project.

Recommended repo name: `HannanSeyfi/unlearning-thesis`.

## Option 1: GitHub Website

1. Create a new GitHub repository named `unlearning-thesis`.
2. Choose public or private.
3. Upload the contents of this `github-ready` folder, not the folder itself.
4. Commit the upload to the `main` branch.
5. Open `COLAB_NOTEBOOKS.md` in GitHub and use the Colab links.

## Option 2: Git Command Line

Install Git first, then run these commands inside this `github-ready` folder:

```powershell
git init
git add .
git commit -m "Initial Colab project"
git branch -M main
git remote add origin https://github.com/HannanSeyfi/unlearning-thesis.git
git push -u origin main
```

If you use a different repo name, update these files after publishing:

- `README.md`
- `COLAB_NOTEBOOKS.md`
- `setup_colab_from_github.ipynb`
