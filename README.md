# Research Project Example
Example to show possible approaches for project organization, config handling and experiment tracking for deep learning research.

## Preparations
To use the example project, you should first make the necessary preparations to clone it and install the necessary packages. To make the preparations, you should 

### 0. Clone the Repository
Clone the repository with your preferred method and navigate into it in a terminal.
```bash
git clone https://github.com/VisionLab-IIT/research_project_example.git
```
```bash
cd research_project_example
```
### 1. (optional) Create a Virtual Environment
It is good practice to keep installed packages in separate virtual environments. On Linux, you can create a Python virtual environment with
```bash
python3 -m venv .venv
```
Here, `.venv` will be the directory of your virtual environment. 

### 2. Requirements
You can install the requirements from `requirements.txt` with
```bash
pip install -r requirements.txt
```

# Training the model

### 1. (optional, if venv has been created) Activate Virtual Environment
Before running the training script, activate the environment with
```bash
source .venv/bin/activate
```

### 2. Run Training Script
> [!WARNING]
> By default, the code will automatically download the dataset if it is not present in that location.

To train the example model (which is inspired by the [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) architecture), run
```bash
python3 main_training.py --data_path ./data
```
specifing the location of the training data with the `--data_path` argument. 
> [!TIP]
> You can use a subset of the training data with the `--train_set_ratio` argument.