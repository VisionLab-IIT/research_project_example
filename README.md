# Research Project Example
Example to show possible approaches for project organization, config handling and experiment tracking for deep learning research.

## Preparations
To use the example project, you should first make the necessary preparations to clone it and install the necessary packages.

### 0. Clone the repository
Open a terminal and navigate to the directory where you would like the project to be placed, then
```bash
git clone https://github.com/VisionLab-IIT/research_project_example.git
```
### 1. Change your shell's working directory to the project
```bash
cd research_project_example
```

### 2. (optional) Create a Python virtual environment
It is good practice to keep installed packages of separate projects in their own virtual environments. On Linux, you can create a Python virtual environment with
```bash
python3 -m venv .venv
```
Here, `.venv` will be the directory of your virtual environment. 

### 3. Install requirements
You can install the requirements from `requirements.txt`.

#### 3.1. If you want to install in your virtual environment, activate it first
```bash
source .venv/bin/activate
```
#### 3.2. Then install the requirements
```bash
pip install -r requirements.txt
```

# Checkout the development stage of your choice
You can navigate between the different stages of the project's development. Choose a stage from the list below so you can explore its changes.

Use `git checkout` to switch between stages with either the stage's tag
```bash
git checkout [state_tag]
```
or the corresponding commit hash
```bash
git checkout [commit_hash]
```

Here is the list of the most important stages:

| Stage | Tag | Commit hash (short version) | Description |
|-------|---------------------|------|--------|
| 1. Baseline | 01_baseline | 27659e3 | The first state worth checking out as a starting point. |
| 2. Simple Tracker | 02_simple_tracker | 82a1267 | `ExperimentTracker` class to encapsulate functionalities for logging, comparison and reproducibility. <br>At this stage, only the basics from stage 01 are reorganized here. |
| 3. Log Directories | 02_log_dirs | e5b5cee | Tracking logs into separate directories under log. <br>This enables basic comparisons like checking plots of different runs. |

> [!IMPORTANT]
> From this point, the instructions may differ more between each stage, so please read carefully!
# Training

### 1. If you work in virtual environment, activate it.
Before running the training script, activate the environment with
```bash
source .venv/bin/activate
```

### 2. Start Training
> [!WARNING]
> By default, the code will automatically create the given data path if it does not exist and download the dataset if it is not present in that location.

To train the example model (which is inspired by the [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) architecture), run
```bash
python3 main_training.py --data_path ./data
```
specifying the location of the training data with the `--data_path` argument. 
> [!TIP]
> You can use a subset of the training data with the `--train_set_ratio` argument.