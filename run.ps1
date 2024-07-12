#!/bin/bash

# ConfMask
python .\experiments\gen.py -r 2 -h 2 -s 0 -n A -n D -n E -n G
python .\experiments\gen.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
python .\experiments\gen.py -r 6 -h 4 -s 0 -n A -n B -n C -n D -n E -n G
python .\experiments\gen.py -r 6 -h 6 -s 0 -n A -n B -n C
python .\experiments\gen.py -r 10 -h 2 -s 0 -n A -n D -n E -n G

# Strawmans
python .\experiments\gen.py -r 6 -h 2 -s 0 -n A -n C -n D -n E -n F -n H -a strawman1
python .\experiments\gen.py -r 6 -h 2 -s 0 -n A -n C -n D -n E -n F -n H -a strawman2

# Experiments
python .\experiments\5.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
python .\experiments\5.py -r 2 -h 2 -s 0 -n A -n D -n E -n G
python .\experiments\5.py -r 10 -h 2 -s 0 -n A -n D -n E -n G
python .\experiments\5.py -r 6 -h 4 -s 0 -n A -n B -n C
python .\experiments\5.py -r 6 -h 6 -s 0 -n A -n B -n C
python .\experiments\5.py -r 6 -h 2 -s 0 -n A -a strawman1
python .\experiments\5.py -r 6 -h 2 -s 0 -n A -a strawman2
python .\experiments\6.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
python .\experiments\7.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
python .\experiments\8.py -r 6 -h 2 -s 0 -n A -n D -n G
python .\experiments\9.py -r 6 -h 4 -s 0 -n A -n B -n C -n D -n G
python .\experiments\10.py -r 6 -h 2 -s 0 -n A
python .\experiments\11.py -r 2 -r 6 -r 10 -h 2 -s 0 -n A -n D -n E -n G
python .\experiments\12.py -r 6 -h 2 -h 4 -h 6 -s 0 -n A -n B -n C
python .\experiments\13.py -r 2 -r 6 -r 10 -h 2 -s 0 -n A -n D -n E -n G
python .\experiments\14.py -r 6 -h 2 -h 4 -s 0 -n A -n D -n E -n G
python .\experiments\15.py -s 0 `
    -c 2,2,A -c 2,2,D -c 2,2,E `
    -c 6,2,A -c 6,2,B -c 6,2,C -c 6,2,D -c 6,2,E -c 6,2,G `
    -c 6,4,A -c 6,4,B -c 6,4,C `
    -c 10,2,A -c 10,2,D -c 10,2,E
python .\experiments\16.py -r 6 -h 2 -s 0 -n A -n C -n D -n E -n F -n H