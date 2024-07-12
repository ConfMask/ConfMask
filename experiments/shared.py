"""Shared functionalities for the experiments."""

import shlex

import click
import rich
from rich.progress import Progress, TimeElapsedColumn, TextColumn
from rich.table import Column

from config import ALGORITHMS, AVAIL_NETWORKS


### CLI options ###


def cli_network(multiple=False):
    if multiple:
        return click.option(
            "-n",
            "--networks",
            required=True,
            type=click.Choice(sorted(AVAIL_NETWORKS)),
            multiple=True,
            help="Networks to run.",
        )

    return click.option(
        "-n",
        "--network",
        required=True,
        type=click.Choice(sorted(AVAIL_NETWORKS)),
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


### CLI display ###


def display_title(id, **kwargs):
    """Display the title of a script.

    Parameters
    ----------
    id : any
        The identifier of the script.
    kwargs : dict
        Parameters to be displayed in the title. The keys are treated as the parameter
        names.
    """
    params = ", ".join(f"{key}={repr(value)}" for key, value in kwargs.items())
    rich.get_console().rule(f"[bold]Figure {id}[/bold] | {params}", characters="━")


def display_progress(networks, run_network_func, clean_network_func=None, **kwargs):
    """Display the progress panel of running networks.

    Parameters
    ----------
    networks : list
        List of networks to run.
    run_network_func : function
        Function to run for each network. It should take the network name as the first
        positional argument, keyword arguments `progress` and `task`, and any additional
        keyword arguments passed through `kwargs`.
    clean_network_func : function, optional
        Function to clean up the network on error. It should take the network name as
        the first positional argument and any additional keyword arguments passed
        through `kwargs`.
    kwargs : dict
        Additional keyword arguments to pass to the functions.
    """
    if len(networks) == 0:
        return

    with Progress(
        TimeElapsedColumn(),
        TextColumn("{task.fields[network]}", style="bold magenta"),
        TextColumn("{task.description}", table_column=Column(min_width=50)),
        TextColumn("{task.fields[details]}", style="dim white"),
    ) as progress:
        tasks = {
            network: progress.add_task(
                "(queued)", start=False, total=None, network=f"[{network}]", details=""
            )
            for network in networks
        }

        for network in networks:
            task = tasks[network]
            try:
                progress.start_task(task)
                run_network_func(network, progress=progress, task=task, **kwargs)
                progress.stop_task(task)
            except Exception:
                progress.update(task, description="[red]Error")
                progress.stop_task(task)
                if clean_network_func is not None:
                    clean_network_func(network, **kwargs)
                progress.console.print(f"[red]Error in network {network}")
                progress.console.print_exception()


### Missing command hints ###


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
