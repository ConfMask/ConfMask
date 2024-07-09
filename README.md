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

Pull the necessary docker images and run Batfish:

```bash
docker pull batfish/allinone
docker pull ghcr.io/confmask/confmask-config2spec:latest  # Used for Experiment 9 
docker run --name batfish -v batfish-data:/data -p 8888:8888 -p 9997:9997 -p 9996:9996 batfish/allinone
```

## Evaluation

In short, run `./run.sh` on Unix or `./run.ps1` on Windows to run the full evaluation
suite. Read on if you want to look into details, or something broke halfway and you do
not want to start over.

> [!NOTE]
> - Use the `--help` option on each script to see available options.
> - The generation script does not overwrite existing data by default. Use the
>   `-f/--force` to force overwrite existing data instead.
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
# See run.sh for minimum required ones to complete the evaluation suite
python experiments/gen.py --kr 6 --kh 2 --seed 0 -n A
```

Run the strawman algorithms (required for [Figure 10](#figure-10) and
[Figure 16](#figure-16)):

```bash
# See run.sh for minimum required ones to complete the evaluation suite
python experiments/gen.py --kr 6 --kh 2 --seed 0 -n A -a strawman1
python experiments/gen.py --kr 6 --kh 2 --seed 0 -n A -a strawman2
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

> [!NOTE]
> This experiment involves Config2Spec, for which we use a modified version to
> support extracting network specifications of both NetHide and ConfMask for comparison.
> We provide a docker image with all necessary dependencies, so please make sure you have
> pulled the image according to the [setup](#setup).

```bash
python experiments/9.py --kr 6 --kh 2 --seed 0
```

### Figure 10

```bash
python experiments/10.py --kr 6 --kh 2  # TODO
```

### Figure 11

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5).

```bash
python ./experiments/11.py --kr 2 --kr 6 --kr 10 --kh 2 --seed 0
```

### Figure 12

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5).

```bash
python ./experiments/12.py --kr 6 --kh 2 --kh 4 --kh 6 --seed 0
```

### Figure 13

```bash
python ./experiments/13.py --kr 2 --kr 6 --kr 10 --kh 2 --seed 0
```

### Figure 14

```bash
python ./experiments/14.py --kr 6 --kh 2 --kh 4 --seed 0
```

### Figure 15

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5). Moreover, comparing
> across different networks and different sets of parameters may not imply strong
> correlation; try controlling variables instead.

```bash
python experiments/15.py --seed 0 \
    -c 2,2,A -c 2,2,D -c 2,2,E \
    -c 6,2,A -c 6,2,B -c 6,2,C -c 6,2,D -c 6,2,E -c 6,2,G \
    -c 6,4,A -c 6,4,B -c 6,4,C \
    -c 10,2,A -c 10,2,D -c 10,2,E
```

### Figure 16

```bash
python experiments/16.py --kr 6 --kh 2  # TODO
```
