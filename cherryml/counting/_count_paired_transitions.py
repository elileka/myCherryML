# cherryml/counting/_count_paired_transitions.py
import logging
import multiprocessing
import os
import sys
import tempfile
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

from cherryml import caching
from cherryml.io import (
    read_msa,
    read_site_rates,
    read_tree,
    write_count_matrices,
)
from cherryml.utils import get_process_args, quantization_idx, get_amino_acids

def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

_init_logger()


def _map_func(args) -> List[Tuple[float, pd.DataFrame]]:
    """
    Per-process worker for paired AA+3Di counting.
    Expects args to contain:
      tree_dir, msa_dir_aa, msa_dir_3di, site_rates_dir, families,
      aa_alphabet, tri_alphabet, quantization_points, edge_or_cherry
    """
    (
        tree_dir,
        msa_dir_aa,
        msa_dir_3di,
        site_rates_dir,
        families,
        aa_alphabet,
        tri_alphabet,
        quantization_points,
        edge_or_cherry,
    ) = args

    quantization_points = np.array(sorted(quantization_points))
    num_aa = len(aa_alphabet)
    num_tri = len(tri_alphabet)
    num_states = num_aa * num_tri

    # build maps
    aa_to_idx = {aa: i for (i, aa) in enumerate(aa_alphabet)}
    tri_to_idx = {t: i for (i, t) in enumerate(tri_alphabet)}
    # state labels in same ordering as index: [for aa in aa_alphabet: for tri in tri_alphabet]
    states = [f"{aa}|{tri}" for aa in aa_alphabet for tri in tri_alphabet]

    # shape: (num_quantization_points, num_states, num_states)
    count_matrices_numpy = np.zeros(
        shape=(len(quantization_points), num_states, num_states)
    )

    for family in families:
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        msa_aa = read_msa(msa_path=os.path.join(msa_dir_aa, family + ".txt"))
        msa_tri = read_msa(msa_path=os.path.join(msa_dir_3di, family + ".txt"))
        site_rates = read_site_rates(
            site_rates_path=os.path.join(site_rates_dir, family + ".txt")
        )

        # Basic consistency check: same keys and lengths
        # (Assume read_msa returns dict-like keyed by leaf names / node ids)
        # If MSAs are arrays/lists keyed by node index, adapt accordingly.
        # Here we assume same keys/order as in _count_transitions.py
        if set(msa_aa.keys()) != set(msa_tri.keys()):
            raise ValueError(
                f"AA and 3Di MSAs differ in sequence headers for family {family}"
            )

        if edge_or_cherry == "cherry++":
            total_pairs = []

            def dfs(node):
                if tree.is_leaf(node):
                    return (node, 0.0)
                unmatched_leaves_under = []
                distances_under = []
                for child, branch_length in tree.children(node):
                    maybe_unmatched_leaf, maybe_distance = dfs(child)
                    if maybe_unmatched_leaf is not None:
                        assert maybe_distance is not None
                        unmatched_leaves_under.append(maybe_unmatched_leaf)
                        distances_under.append(maybe_distance + branch_length)
                assert len(unmatched_leaves_under) == len(distances_under)
                index = 0
                while index + 1 <= len(unmatched_leaves_under) - 1:
                    total_pairs.append(1)
                    (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                        (unmatched_leaves_under[index], distances_under[index]),
                        (unmatched_leaves_under[index + 1], distances_under[index + 1]),
                    )
                    leaf_seq_1_aa, leaf_seq_1_tri = msa_aa[leaf_1], msa_tri[leaf_1]
                    leaf_seq_2_aa, leaf_seq_2_tri = msa_aa[leaf_2], msa_tri[leaf_2]
                    msa_length = len(leaf_seq_1_aa)
                    for pos in range(msa_length):
                        site_rate = site_rates[pos]
                        branch_length_total = branch_length_1 + branch_length_2
                        q_idx = quantization_idx(branch_length_total * site_rate, quantization_points)
                        if q_idx is not None:
                            aa1 = leaf_seq_1_aa[pos]
                            aa2 = leaf_seq_2_aa[pos]
                            tri1 = leaf_seq_1_tri[pos]
                            tri2 = leaf_seq_2_tri[pos]
                            # require both characters to be valid (no gaps)
                            if aa1 in aa_alphabet and aa2 in aa_alphabet and tri1 in tri_alphabet and tri2 in tri_alphabet:
                                state1_idx = aa_to_idx[aa1] * num_tri + tri_to_idx[tri1]
                                state2_idx = aa_to_idx[aa2] * num_tri + tri_to_idx[tri2]
                                # cherries count as 0.5 to each direction (symmetric)
                                count_matrices_numpy[q_idx, state1_idx, state2_idx] += 0.5
                                count_matrices_numpy[q_idx, state2_idx, state1_idx] += 0.5
                    index += 2
                if len(unmatched_leaves_under) % 2 == 0:
                    return (None, None)
                else:
                    return (unmatched_leaves_under[-1], distances_under[-1])

            dfs(tree.root())
            # nothing else to do for cherry++
        else:
            for node in tree.nodes():
                if edge_or_cherry == "edge":
                    node_seq_aa = msa_aa[node]
                    node_seq_tri = msa_tri[node]
                    msa_length = len(node_seq_aa)
                    for child, branch_length in tree.children(node):
                        child_seq_aa = msa_aa[child]
                        child_seq_tri = msa_tri[child]
                        for pos in range(msa_length):
                            site_rate = site_rates[pos]
                            q_idx = quantization_idx(branch_length * site_rate, quantization_points)
                            if q_idx is not None:
                                aa_start = node_seq_aa[pos]
                                tri_start = node_seq_tri[pos]
                                aa_end = child_seq_aa[pos]
                                tri_end = child_seq_tri[pos]
                                if (
                                    aa_start in aa_alphabet
                                    and tri_start in tri_alphabet
                                    and aa_end in aa_alphabet
                                    and tri_end in tri_alphabet
                                ):
                                    start_idx = aa_to_idx[aa_start] * num_tri + tri_to_idx[tri_start]
                                    end_idx = aa_to_idx[aa_end] * num_tri + tri_to_idx[tri_end]
                                    count_matrices_numpy[q_idx, start_idx, end_idx] += 1
                elif edge_or_cherry == "cherry":
                    children = tree.children(node)
                    if len(children) == 2 and all([tree.is_leaf(child) for (child, _) in children]):
                        (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (children[0], children[1])
                        leaf_seq_1_aa, leaf_seq_1_tri = msa_aa[leaf_1], msa_tri[leaf_1]
                        leaf_seq_2_aa, leaf_seq_2_tri = msa_aa[leaf_2], msa_tri[leaf_2]
                        msa_length = len(leaf_seq_1_aa)
                        for pos in range(msa_length):
                            site_rate = site_rates[pos]
                            branch_length_total = branch_length_1 + branch_length_2
                            q_idx = quantization_idx(branch_length_total * site_rate, quantization_points)
                            if q_idx is not None:
                                aa1 = leaf_seq_1_aa[pos]
                                aa2 = leaf_seq_2_aa[pos]
                                tri1 = leaf_seq_1_tri[pos]
                                tri2 = leaf_seq_2_tri[pos]
                                if aa1 in aa_alphabet and aa2 in aa_alphabet and tri1 in tri_alphabet and tri2 in tri_alphabet:
                                    state1_idx = aa_to_idx[aa1] * num_tri + tri_to_idx[tri1]
                                    state2_idx = aa_to_idx[aa2] * num_tri + tri_to_idx[tri2]
                                    count_matrices_numpy[q_idx, state1_idx, state2_idx] += 0.5
                                    count_matrices_numpy[q_idx, state2_idx, state1_idx] += 0.5

    # Build output list of (quant_point, DataFrame) matching write_count_matrices format
    count_matrices = [
        [
            q,
            pd.DataFrame(
                count_matrices_numpy[q_idx, :, :],
                index=states,
                columns=states,
            ),
        ]
        for (q_idx, q) in enumerate(quantization_points)
    ]
    return count_matrices


@caching.cached_computation(
    exclude_args=[
        "num_processes",
        "use_cpp_implementation",
        "cpp_command_line_prefix",
        "cpp_command_line_suffix",
    ],
    output_dirs=["output_count_matrices_dir"],
    write_extra_log_files=True,
)
def count_paired_transitions(
    tree_dir: str,
    msa_dir_aa: str,
    msa_dir_3di: str,
    site_rates_dir: str,
    families: List[str],
    tri_alphabet: List[str],
    aa_alphabet: Optional[List[str]] = None,
    quantization_points: List[Union[str, float]] = None,
    edge_or_cherry: str = "cherry++",
    output_count_matrices_dir: Optional[str] = None,
    num_processes: int = 1,
    use_cpp_implementation: bool = False,
    cpp_command_line_prefix: str = "",
    cpp_command_line_suffix: str = "",
) -> None:
    """
    Count transitions over the product alphabet (AA,3Di) collapsed to single states.

    Args are analogous to count_transitions but we accept two msa directories and tri_alphabet.
    """
    start_time = time.time()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting paired counting on {len(families)} families")

    if aa_alphabet is None:
        aa_alphabet = get_amino_acids()

    if quantization_points is None:
        raise ValueError("quantization_points must be provided")

    if not os.path.exists(output_count_matrices_dir):
        os.makedirs(output_count_matrices_dir)
    quantization_points = [float(q) for q in quantization_points]

    # For now we do not provide C++ implementation. Keep Python path only.
    map_args = [
        [
            tree_dir,
            msa_dir_aa,
            msa_dir_3di,
            site_rates_dir,
            get_process_args(process_rank, num_processes, families),
            aa_alphabet,
            tri_alphabet,
            quantization_points,
            edge_or_cherry,
        ]
        for process_rank in range(num_processes)
    ]

    # Map step
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            count_matrices_per_process = list(
                tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args))
            )
    else:
        count_matrices_per_process = list(
            tqdm.tqdm(map(_map_func, map_args), total=len(map_args))
        )

    # Reduce step
    count_matrices = count_matrices_per_process[0]
    for process_rank in range(1, num_processes):
        for q_idx in range(len(quantization_points)):
            count_matrices[q_idx][1] += count_matrices_per_process[process_rank][q_idx][1]

    write_count_matrices(count_matrices, os.path.join(output_count_matrices_dir, "result.txt"))

    logger.info("Done!")
    with open(os.path.join(output_count_matrices_dir, "profiling.txt"), "w") as profiling_file:
        profiling_file.write(
            f"Total time: {time.time() - start_time} seconds with {num_processes} processes.\n"
        )
