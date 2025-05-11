# pico-llm
Machine Learning project

1. Install dependencies from requirements.txt file. Make sure to create a virtual/conda environment before running this command.

```
# create new env called pico_env
conda create -n pico_env python=3.11

# activate pico_env
conda activate pico_env

# install dependencies
pip install -r requirements.txt

```

2. Run `main.py` file. Change `config-name` to train different models

```
cd src
python main.py --config-path=config --config-name=train_lstm
```

Note: Config files can be found in the `configs` folders. No need for command-line arguments.
