# Network A
python .\experiments\gen.py --kr 2 --kh 2 --seed 0 -n A
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n A
python .\experiments\gen.py --kr 6 --kh 4 --seed 0 -n A
python .\experiments\gen.py --kr 6 --kh 6 --seed 0 -n A
python .\experiments\gen.py --kr 10 --kh 2 --seed 0 -n A

# Network B
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n B
python .\experiments\gen.py --kr 6 --kh 4 --seed 0 -n B
python .\experiments\gen.py --kr 6 --kh 6 --seed 0 -n B

# Network C
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n C
python .\experiments\gen.py --kr 6 --kh 4 --seed 0 -n C
python .\experiments\gen.py --kr 6 --kh 6 --seed 0 -n C

# Network D
python .\experiments\gen.py --kr 2 --kh 2 --seed 0 -n D
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n D
python .\experiments\gen.py --kr 6 --kh 4 --seed 0 -n D
python .\experiments\gen.py --kr 10 --kh 2 --seed 0 -n D

# Network E
python .\experiments\gen.py --kr 2 --kh 2 --seed 0 -n E
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n E
python .\experiments\gen.py --kr 6 --kh 4 --seed 0 -n E
python .\experiments\gen.py --kr 10 --kh 2 --seed 0 -n E

# Network F
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n F

# Network G
python .\experiments\gen.py --kr 2 --kh 2 --seed 0 -n G
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n G
python .\experiments\gen.py --kr 6 --kh 4 --seed 0 -n G
python .\experiments\gen.py --kr 10 --kh 2 --seed 0 -n G

# Network H
python .\experiments\gen.py --kr 2 --kh 2 --seed 0 -n H
python .\experiments\gen.py --kr 6 --kh 2 --seed 0 -n H
python .\experiments\gen.py --kr 10 --kh 2 --seed 0 -n H

# Experiments
python .\experiments\5.py --kr 6 --kh 2 --seed 0
python .\experiments\5.py --kr 2 --kh 2 --seed 0  # For 11
python .\experiments\5.py --kr 10 --kh 2 --seed 0  # For 11
python .\experiments\5.py --kr 6 --kh 4 --seed 0  # For 12
python .\experiments\5.py --kr 6 --kh 6 --seed 0  # For 12
python .\experiments\6.py --kr 6 --kh 2 --seed 0
python .\experiments\7.py --kr 6 --kh 2 --seed 0
python .\experiments\11.py --kr 2 --kr 6 --kr 10 --kh 2 --seed 0
python .\experiments\12.py --kr 6 --kh 2 --kh 4 --kh 6 --seed 0
