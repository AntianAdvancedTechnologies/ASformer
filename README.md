# ASformer
## PYTHON

Download Python 3.7.9 and create a venv

`python -m venv venv`

## Requirements

Download packages using requirements.txt

`pip install -r requirements.txt`


# Project Directory Structure

The project contains the following main directories:

- `data/`: This directory contains all the datasets used in the project.
    - `snatch/`: This sub-directory contains the dataset related to the 'Snatch' weightlifting technique.
        - `features/`: This sub-directory contains the features dataset for 'Snatch'.
    - `cleanjerk/`: This sub-directory contains the dataset related to the 'Clean and Jerk' weightlifting technique.
        - `features/`: This sub-directory contains the features dataset for 'Clean and Jerk'.

# To run
`
python Asformer/main.py --action=train --dataset=snatch --split=1/2/3/4/5
`

# To train
`
python main.py --action=train --dataset=snatch
`


# To predict
`
python main.py --action=predict --dataset=snatch
`


# To test
`
python main.py --action=test --dataset=snatch
`


# To eval
`
python eval.py --dataset=snatch
`
