#!/bin/bash

floyd run --gpu --data nikulaj/datasets/mnist:data --env keras "python train_and_eval.py"
