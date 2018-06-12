# OpenAI-gym-MountainCar-v0
use seq2seq to implement simple arithmetic operation

## files & directories
- fig/*: the figures of each experiment
- console_log/*: the console output of each experiment
- hw4.py

## requirements

- python3.6
- tensorflow
- more details are writed as a `requirements.txt` file

## usage

- run an experiments
    - use MODEL_NAME to specify size of model name
    - use TRAIN_EPISODE to specify train times
    - use STEP_LIMIT to specify the maximum possible action step in each episode
    - use STATE_NUM to specify the state number size of Q-matrix
    - use LEARNING_RATE to specify the learning rate
```sh
$ python3 hw4.py [-h] [-n MODEL_NAME] [-e TRAIN_EPISODE] [-l STEP_LIMIT] [-s STATE_NUM] [-lr LEARNING_RATE]
```

- You can use shell script to execute the models

```sh
$ bash command.sh
```

## detail and report

You can refer to jupyter notebook [report.ipynb](https://nbviewer.jupyter.org/github/rapirent/DSAI-HW4/blob/master/report.ipynb)

## LICNESE

MIT
