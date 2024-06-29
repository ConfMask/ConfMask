#!/bin/bash

python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n A
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n B
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n C
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n D
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n E
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n F
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n G
python ./experiments/gen.py --kr 6 --kh 2 --seed 0 -n H

python ./experiments/5.py --kr 6 --kh 2 --seed 0
python ./experiments/6.py --kr 6 --kh 2 --seed 0
