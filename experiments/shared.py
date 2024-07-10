import shlex

import click

from config import PROTOCOL_MAPPING, ALGORITHMS


def cli_network(multiple=False):
    if multiple:
        return click.option(
            "-n",
            "--networks",
            required=True,
            type=click.Choice(sorted(PROTOCOL_MAPPING)),
            multiple=True,
            help="Networks to run.",
        )

    return click.option(
        "-n",
        "--network",
        required=True,
        type=click.Choice(sorted(PROTOCOL_MAPPING)),
        help="Network to run.",
    )


def cli_algorithm():
    return click.option(
        "-a",
        "--algorithm",
        type=click.Choice(ALGORITHMS),
        default="confmask",
        help="Algorithm to run.",
    )


def cli_kr(multiple=False):
    if multiple:
        return click.option(
            "-r",
            "--krs",
            required=True,
            type=int,
            multiple=True,
            help="Router anonymization degrees.",
        )

    return click.option(
        "-r",
        "--kr",
        required=True,
        type=int,
        help="Router anonymization degree.",
    )


def cli_kh(multiple=False):
    if multiple:
        return click.option(
            "-h",
            "--khs",
            required=True,
            type=int,
            multiple=True,
            help="Host anonymization degrees.",
        )

    return click.option(
        "-h",
        "--kh",
        required=True,
        type=int,
        help="Host anonymization degree.",
    )


def cli_seed():
    return click.option(
        "-s",
        "--seed",
        required=True,
        type=int,
        help="Random seed.",
    )


def cli_force_overwrite():
    return click.option(
        "-f",
        "--force-overwrite",
        is_flag=True,
        help="Force overwrite existing data.",
    )


def cli_plot_only():
    return click.option(
        "-p",
        "--plot-only",
        is_flag=True,
        help="Plot based on stored results without running any evaluation.",
    )


def get_gen_cmd(network, algorithm, kr, kh, seed):
    parts = [
        "python",
        "./experiments/gen.py",
        "-r",
        str(kr),
        "-h",
        str(kh),
        "-s",
        str(seed),
        "-n",
        network,
    ]

    if algorithm != "confmask":
        parts.append("-a")
        parts.append(algorithm)

    return shlex.join(parts)


def get_5_cmd(networks, algorithm, kr, kh, seed):
    parts = [
        "python",
        "./experiments/5.py",
        "-r",
        str(kr),
        "-h",
        str(kh),
        "-s",
        str(seed),
    ]

    for network in networks:
        parts.append("-n")
        parts.append(network)

    if algorithm != "confmask":
        parts.append("-a")
        parts.append(algorithm)

    return shlex.join(parts)
