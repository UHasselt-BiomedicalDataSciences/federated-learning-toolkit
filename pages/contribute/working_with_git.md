---
title: Working with Git
---

This guide explains how to request and edit a page using Git and GitHub.

## Prerequisites

- Basic knowledge of Markdown. Refer to our [Markdown cheat sheet](markdown_cheat_sheet).
- Basic Git skills.
- A GitHub account. If you don't have one, [create a free GitHub account](https://github.com/join) before proceeding.

---

## Workflow: Fork → Branch → Change → Push → PR

This is a general workflow in how to work on your own fork (copy) of the `federated-learning-toolkit` repo and request
changes through a pull request:

> NOTE: if you already did these steps in the past, start from the `git fetch upstream` command.

- Make a fork of this repository, using the fork button on the home page .

- Forking the repository lets you preview your own deployed version of the site.
  This is useful for both reviewing your work and sharing a live preview with others during pull requests.

    - On your fork’s GitHub page, look at the **About** section in the top-right. Click the **gear icon (⚙️)** to open its settings.  
      Enable **"Use your GitHub Pages website"** — this will update the website link to point to your fork’s deployment.

    - Then go to **Settings → Environments → github-pages**, and under **Deployment branches and tags**, remove the `main` restriction.  
      This allows GitHub Pages to deploy from any branch — not just `main`. Alternatively choose `"No Restrictions"` from a dropdown.


- Open a terminal and clone your fork (replace `<USERNAME>` with your GitHub username):
    ```bash
    git clone git@github.com:<USERNAME>/federated-learning-toolkit.git
    cd federated-learning-toolkit
    ```
    > NOTE: Make sure you clone the fork and not the original UHasselt-BiomedicalDataSciences/federated-learning-toolkit one.

- Keep your fork up to date (IMPORTANT!) with original ("upstream") repo.
    ```bash
    # Add the original repo as a remote called "upstream"
    git remote add upstream {{site.REPO}}.git

    # Fetch the latest changes from upstream
    git fetch upstream

    # Switch to your local main branch (check with `git branch` if unsure)
    git checkout main

    # Merge the latest changes from upstream/main into your local main
    git pull upstream main
    ```

- Create a new branch to which you will push changes. Name it after your feature/edit.
    ```bash
    git checkout -b '<FEATURE_NAME>'
    ```

- Make the changes you want to make using an editor of choice

- Save changes and push

    - Open terminal and stage your changes:
        ```bash
        git add .
        ```
  - Committing changes
      ```bash
      git commit -m "Changing the tool-resource file"
      ```
  - Pushing you changes to your fork
      ```bash
      git push origin '<FEATURE_NAME>'
      ```
- The act of pushing new changes repo (fork) have triggered `Jekyll site CI` action which has `build` and `deploy` jobs.
  By going to "Actions" tab we can see a list of recently executed workflows.
  Once the latest has succesfully deployed we can see the result at: `https://<USERNAME>.github.io/federated-learning-toolkit/`.
  
  > Note: You will most likely have to hard-refresh the browser page to get latest changes.
   
  Now that we can successfully demonstrate how our changes look like it's time to create a Pull Request.

- Go to the **upstream repository**: <{{site.REPO}}> and click **Compare & pull request**.
  You’re proposing to merge changes from your fork’s branch (`FEATURE_NAME`) into the `main` branch of the original repository.
- Open the pull request and briefly describe what you changed and why.
- Wait for the review process. Editors responsible for the sections you modified will be automatically assigned as reviewers.

## The advantage of working locally: previewing your changes through your web browser

The website is build on GitHub using Jekyll, a simple, static site generator based on ruby. When you have a local copy
cloned onto your computer, it is possible to generate the website based on this repo. This makes it possible to preview
changes live, every time you save a file from within the GitHub federated-learning-toolkit repo. Follow these steps to deploy the website
based on your local clone (copy) of the  repo:

Make sure you have cloned the federated-learning-toolkit repo:
```bash
    git clone git@github.com:<USERNAME>/federated-learning-toolkit.git
    cd federated-learning-toolkit
```

To run the website locally, you can either use {% tool "docker" %} or use Jekyll directly after installing various dependencies.

### Run using Jekyll directly


It's recommended to use **Ruby 3.1**, as this is the version used by GitHub Pages, and we know newer version can be incompatible with current dependencies.

#### 1. Install Ruby

On Mac a quick solution is to install it via homebrew:
```bash
brew install ruby@3.1
```
Then update your PATH env variable with: `export PATH="/opt/homebrew/opt/ruby@3.1/bin:$PATH"`

> Note: If you want to manage several ruby versions or need to install on Linux/Windows consult the: [https://jekyllrb.com/docs/installation/](https://jekyllrb.com/docs/installation/)


#### 2. Install Dependencies & Run the Server
1. Mac/Linux:
   In the terminal go to `federated-learning-toolkit`.
   Run `gem install bundler` (needed only once). Then run `make install`.
   Now you can run `make dev` to start the server. The local deployment should be available at [http://localhost:4000/](http://localhost:4000/).
   Any change to the code will automatically be reflected in the browser.

2. Windows: We recommend using **Bash** and `make` whenever possible. You can do this via:
   - **Git Bash** (installed with [Git for Windows](https://git-scm.com/))
   - **WSL (Windows Subsystem for Linux)** with a Linux distribution like Ubuntu

   If you prefer using **Windows PowerShell**, you can still run the Jekyll development server manually:
   ```powershell
   # Run once to install dependencies
   gem install bundler
   bundle install

   # Set environment variables and start the server
   $env:JEKYLL_BUILD_BRANCH = git rev-parse --abbrev-ref HEAD
   $env:JEKYLL_ENV = "development"
   bundle exec jekyll serve --livereload --incremental --trace
   ```

### Run using Docker

> Note: this does not run on macs with Apple Silicon (and it hasn't been tested elsewhere).

If not already installed on your machine, install Docker.
From the root of the ``federated-learning-toolkit`` directory, run:
- Linux/Windows
  ```bash
  docker run -it --rm -p 4000:4000 -v ${PWD}:/srv/jekyll jekyll/jekyll:4 /bin/bash -c "chmod -R 777 /srv/jekyll && bundle install && bundle exec jekyll serve --livereload --incremental --trace --host 0.0.0.0"
  ```
- Mac:
  `jekyll:4` image is not supported. An Apple Silicon equivalent that has correct ruby 3.1 doesn't exist...

This will start the docker container and serve the website locally 🤞 (similar to previous example).
