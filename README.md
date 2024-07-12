# ConfMask

This repository contains the source code and evaluation scripts for the paper
[*ConfMask: Enabling Privacy-Preserving Configuration Sharing via Anonymization*](./paper.pdf).

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

Pull and start Batfish service:

```bash
docker pull batfish/allinone
docker run --name batfish -v batfish-data:/data -p 8888:8888 -p 9997:9997 -p 9996:9996 batfish/allinone
```

Pull [confmask-config2spec](https://github.com/orgs/confmask/packages/container/package/confmask-config2spec)
which will be used in [Figure 9](#figure-9):

```bash
docker pull ghcr.io/confmask/confmask-config2spec:latest
```

> [!IMPORTANT]
> We suggest running on machine or server with at least 32GB of memory, otherwise the
> Batfish service may run out of memory for large networks (e.g. Network F).

## Evaluation

Run [run.sh](./run.sh) on Unix or [run.ps1](./run.ps1) on Windows to run the full
evaluation suite. If anything fails during the run, simply re-run the script and
completed generation and evaluation tasks will be automatically skipped.

> [!TIP]
> Run in a terminal window of at least 75 characters wide to enable full display of the
> execution progress.

## Evaluation Details

This section describes each script in more details.

> [!TIP]
> - Use the `--help` option on each script to see available options. An option can be
>   specified multiple times if it is in plural form.
> - Scripts with the `-f/--force-overwrite` option do not overwrite existing data by
>   default. Set the flag to overwrite instead.
> - Evaluation scripts with the `-p/--plot-only` option allows generating plots only
>   from existing data, without running any additional experiments.

### Generate anonymized networks

The [gen.py](./experiments/gen.py) script is used for generating anonymized networks,
with either the ConfMask algorithm or the strawman algorithms mentioned in the paper.

- Use `-n/--networks` to select the networks to run. It can be used multiple times so
  as to anonymize multiple networks with the same setup.
- Use `-a/--algorithm` to select the algorithm to use.
- Use `-r/--kr` and `-h/--kh` to specify the anonymization degrees.
- Use `-s/--seed` to specify a particular random seed.

### Figure 5

```bash
python ./experiments/5.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
```

### Figure 6

```bash
python ./experiments/6.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
```

### Figure 7

```bash
python ./experiments/7.py -r 6 -h 2 -s 0 -n A -n B -n C -n D -n E -n F -n G -n H
```

### Figure 8

> [!NOTE]
> This experiment involves NetHide[^1], thus only a subset of networks is supported.
> Also note that ConfMask should reach a theoretical 100% in this experiment but this
> may not always be the case with this script[^2].

```bash
python ./experiments/8.py -r 6 -h 2 -s 0 -n A -n D -n G
```

### Figure 9

> [!NOTE]
> This experiment involves NetHide[^1] and Config2Spec[^3], thus only a subset of
> networks is supported.

```bash
python ./experiments/9.py -r 6 -h 4 -s 0 -n A -n B -n C -n D -n G
```

### Figure 10

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5). It supports selecting
> only one network at a time.

```bash
python ./experiments/10.py -r 6 -h 2 -s 0 -n A
```

### Figure 11

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5). It supports selecting
> multiple `-r/--krs` values for comparison.

```bash
python ./experiments/11.py -r 2 -r 6 -r 10 -h 2 -s 0 -n A -n D -n E -n G
```

### Figure 12

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5). It supports selecting
> multiple `-h/--khs` values for comparison.

```bash
python ./experiments/12.py -r 6 -h 2 -h 4 -h 6 -s 0 -n A -n B -n C
```

### Figure 13

> [!NOTE]
> This experiment supports selecting multiple `-r/--krs` values for comparison.

```bash
python ./experiments/13.py -r 2 -r 6 -r 10 -h 2 -s 0 -n A -n D -n E -n G
```

### Figure 14

> [!NOTE]
> This experiment supports selecting multiple `-h/--khs` values for comparison.

```bash
python ./experiments/14.py -r 6 -h 2 -h 4 -s 0 -n A -n D -n E -n G
```

### Figure 15

> [!NOTE]
> This experiment relies on the results of [Figure 5](#figure-5). It uses `-c/--cases`
> to select multiple network and parameter combinations to plot, different from other
> scripts.

> [!NOTE]
> Comparing across different networks and different sets of parameters as in the paper
> may not imply strong correlation; try controlling variables instead.

```bash
python ./experiments/15.py -s 0 \
    -c 2,2,A -c 2,2,D -c 2,2,E \
    -c 6,2,A -c 6,2,B -c 6,2,C -c 6,2,D -c 6,2,E -c 6,2,G \
    -c 6,4,A -c 6,4,B -c 6,4,C \
    -c 10,2,A -c 10,2,D -c 10,2,E
```

### Figure 16

> [!NOTE]
> Running time may vary on different devices, especially since the algorithms are
> parallelized on all available CPU cores. It only makes sense to compare relatively
> the running time of different algorithms.

```bash
python ./experiments/16.py -r 6 -h 2 -s 0 -n A -n C -n D -n E -n F -n H
```

[^1]: [NetHide](https://www.usenix.org/conference/usenixsecurity18/presentation/meier)
is not open-source, thus our re-implementation in [nethide.py](./confmask/nethide.py).
We directly store forwarding information in the `nethide/` directory of each network
that supports NetHide evaluation to avoid the complicated setup of the Gurobi optimizer
that it requires.

[^2]: The reason for ConfMask not reaching the theoretical 100% might be some Batfish
traceroute issue. There are several ways to validate that ConfMask reaches 100% in
correspondence to the theoretical proof provided in the paper:
- In the `_diff_routes` function in [gen.py](./gen.py), print out if any `next_hop` in
  `h_rib_new` is not in `h_nh_old`. Validate that nothing is printed out in the last
  iteration.
- In the `_compare_with_origin` function in [8.py](./8.py), print out unmatched routes.
  Then run `traceroute` manually between the source-destination pair in the original and
  the anonymized networks, respectively. It should turn out that they actually match.

[^3]: [Config2Spec](https://www.usenix.org/conference/nsdi20/presentation/birkner) is
[open-source](https://github.com/nsg-ethz/config2spec). We use a slightly modified
version to extract network specifications of ConfMask and NetHide for comparison. See
[setup](#setup) for the Docker image of our version.

