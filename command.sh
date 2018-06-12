#!/bin/bash
python3 ./hw4.py "--model_name=learning_rate0.01" "--learning_rate=0.01"
python3 ./hw4.py "--model_name=learning_rate0.05" "--learning_rate=0.05"
python3 ./hw4.py "--model_name=learning_rate0.1" "--laarning_rate=0.1"
python3 ./hw4.py "--model_name=state5" "--state_num=5"
python3 ./hw4.py "--model_name=state10" "--state_num=10"
python3 ./hw4.py "--model_name=state15" "--state_num=15"
python3 ./hw4.py "--model_name=state20" "--state_num=20"
