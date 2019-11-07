#!/bin/sh
python main.py --lr=0.001 --save='./model/model0.001'
wait
python main.py --lr=0.003 --save='./model/model0.003'
wait
python main.py --lr=0.005 --save='./model/model0.005'
wait
python main.py --lr=0.007 --save='./model/model0.007'
wait
python main.py --lr=0.0001 --save='./model/model0.0001'
wait
python main.py --lr=0.0003 --save='./model/model0.0003'
wait
python main.py --lr=0.0005 --save='./model/model0.0005'
wait
python main.py --lr=0.0007 --save='./model/model0.0007'
wait