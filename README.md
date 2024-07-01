# ConfMask

This repository contains the source code and evaluation scripts for the paper
*ConfMask:Enabling Privacy-Preserving Configuration Sharing via Anonymization*.

**ACM Reference format:**

Yuejie Wang, Qiutong Men, Yao Xiao, Yongting Chen, and Guyue Liu. 2024. ConfMask:
Enabling Privacy-Preserving Configuration Sharing via Anonymization. In
*ACM SIGCOMM 2024 Conference (ACM SIGCOMM ’24), August 4–8, 2024, Sydney, NSW, Australia*.
ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3651890.3672217

## Setup

Make sure the following are available:

- Docker
- Python (>=3.9)

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install necessary dependencies and build editable `confmask`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Pull the docker image and run Batfish:

```bash
docker pull batfish/allinone
docker run --name batfish -v batfish-data:/data -p 8888:8888 -p 9997:9997 -p 9996:9996 batfish/allinone
```

## Evaluation

In short, run `./run.sh` on Unix or `./run.ps1` on Windows to run the full evaluation
suite. Read on if you want to look into details, or something broke halfway and you do
not want to start over.

> [!NOTE]
> - Use the `--help` option on each script to see available options (including available
>   network names).
> - Each evaluation script supports evaluating only a subset of all relevant networks.
>   Use the `--help` option to see relevant network names and use the `-n/--networks`
>   option to specify the subset.
> - Each evaluation script saves/updates the results in the corresponding JSON file, in
>   addition to producing the plots. Hence, the `--plot-only` option can be used to
>   generate the plots from the existing results without re-running the experiments.
> - The layout of the generated plots may be different from the paper, but they convey
>   essentially the same information.

### Generate anonymized networks

Run the ConfMask algorithm (required for all evaluations):

```bash
# Repeat for `-n A` through `-n H`
python experiments/gen.py --kr 6 --kh 2 --seed 0 -n A
```

Run the strawman algorithms (required for [Figure 10](#figure-10) and
[Figure 16](#figure-16)):

```bash
# TODO
```

### Figure 5

> [!NOTE]
> The random noise generator has significant impact on the results of this experiment.
> The results may thus vary with different random seeds, but the overall average should
> be close to the results in the paper.

```bash
python experiments/5.py --kr 6 --kh 2 --seed 0
```

### Figure 6

```bash
python experiments/6.py --kr 6 --kh 2 --seed 0
```

### Figure 7

```bash
python experiments/7.py --kr 6 --kh 2 --seed 0
```

### Figure 8

> [!NOTE]
> This experiment involves NetHide, for which we directly provide intermediate results
> produce by our re-implementation [here](./confmask/nethide.py). This experiment is
> supported only for networks A, D, and G as in the paper.

```bash
python experiments/8.py --kr 6 --kh 2 --seed 0
```

### Figure 9

```bash
python experiments/9.py --kr 6 --kh 2 --seed 0
```

### Figure 10

```bash
python experiments/10.py --kr 6 --kh 2  # TODO
```

### Figure 11

```bash
python experiments/11.py --kr 6 --kh 2  # TODO
```

### Figure 12

```bash
python experiments/12.py --kr 6 --kh 2  # TODO
```

### Figure 13

```bash
python experiments/13.py --kr 6 --kh 2  # TODO
```

### Figure 14

```bash
python experiments/14.py --kr 6 --kh 2  # TODO
```

### Figure 15

```bash
python experiments/15.py --kr 6 --kh 2  # TODO
```

### Figure 16

```bash
python experiments/16.py --kr 6 --kh 2  # TODO
```
