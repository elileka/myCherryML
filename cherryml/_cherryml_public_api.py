"""
Public API for CherryML as applied to the LG model, the co-evolution model, and the AA-3Di model.
"""
import logging
import os
import sys
import tempfile
from functools import partial
from typing import List, Optional

from cherryml import caching, utils
from cherryml.estimation_end_to_end import (
    coevolution_end_to_end_with_cherryml_optimizer,
    lg_end_to_end_with_cherryml_optimizer,
    aa3di_end_to_end_with_cherryml_optimizer,
)
from cherryml.io import read_rate_matrix, write_rate_matrix
from cherryml.markov_chain import get_lg_path
from cherryml.phylogeny_estimation import fast_tree, phyml, fast_cherries


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


_init_logger()
logger = logging.getLogger(__name__)


def cherryml_public_api(
    output_path: str,
    model_name: str,
    msa_dir: str,
    contact_map_dir: Optional[str] = None,
    msa_dir_3di: Optional[str] = None,           # NEW: 3Di MSA directory
    tri_alphabet: Optional[List[str]] = None,     # NEW: 3Di alphabet as list of chars
    tri_alphabet_path: Optional[str] = None,      # NEW: path to file with 3Di alphabet
    tree_dir: Optional[str] = None,
    site_rates_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    num_processes_tree_estimation: int = 32,
    num_processes_counting: int = 8,
    num_processes_optimization: int = 2,
    num_rate_categories: int = 20,
    initial_tree_estimator_rate_matrix_path: str = get_lg_path(),
    num_iterations: int = 1,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    optimizer_device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 500,
    minimum_distance_for_nontrivial_contact: int = 7,
    do_adam: bool = True,
    cherryml_type: str = "cherry++",
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    optimizer_initialization: str = "jtt-ipw",
    sites_subset_dir: Optional[str] = None,
    coevolution_mask_path: Optional[str] = None,
    use_maximal_matching: bool = True,
    families: Optional[List[str]] = None,
    tree_estimator_name: str = "FastTree",
) -> str:
    """
    CherryML method applied to the LG model, co-evolution model, or AA-3Di model.

    Args:
        See description of arguments in https://github.com/songlab-cal/CherryML
    """
    if model_name not in ["LG", "co-evolution", "AA-3Di"]:
        raise ValueError('model_name should be "LG", "co-evolution" or "AA-3Di".')

    if cache_dir is None:
        cache_dir = tempfile.TemporaryDirectory()
        logger.info(
            "Cache directory not provided. Will use temporary directory "
            f"{cache_dir} to cache computations."
        )

    caching.set_cache_dir(cache_dir)

    if families is None:
        families = utils.get_families(msa_dir)

    if tree_estimator_name == "FastTree":
        tree_estimator = fast_tree
    elif tree_estimator_name == "PhyML":
        tree_estimator = phyml
    elif tree_estimator_name == "FastCherries":
        tree_estimator = partial(fast_cherries, max_iters=50)
    else:
        raise ValueError(f"Unknown tree_estimator_name: {tree_estimator_name}")

    # -------------------------------------------------------------------------
    # LG MODEL
    # -------------------------------------------------------------------------
    if model_name == "LG":
        outputs = lg_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_dir,
            families=families,
            tree_estimator=partial(tree_estimator, num_rate_categories=num_rate_categories),
            initial_tree_estimator_rate_matrix_path=initial_tree_estimator_rate_matrix_path,
            num_iterations=num_iterations,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            use_cpp_counting_implementation=use_cpp_counting_implementation,
            optimizer_device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            edge_or_cherry=cherryml_type,
            cpp_counting_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_counting_command_line_suffix=cpp_counting_command_line_suffix,
            num_processes_tree_estimation=num_processes_tree_estimation,
            num_processes_counting=num_processes_counting,
            num_processes_optimization=num_processes_optimization,
            optimizer_initialization=optimizer_initialization,
            sites_subset_dir=sites_subset_dir,
            tree_dir=tree_dir,
            site_rates_dir=site_rates_dir,
        )

    # -------------------------------------------------------------------------
    # CO-EVOLUTION MODEL
    # -------------------------------------------------------------------------
    elif model_name == "co-evolution":
        if num_iterations > 1:
            raise ValueError(
                "Iteration is not used for learning a coevolution model. "
                f"You provided: num_iterations={num_iterations}. Set this "
                "argument to 1 and retry."
            )

        outputs = coevolution_end_to_end_with_cherryml_optimizer(
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            coevolution_mask_path=coevolution_mask_path,
            families=families,
            tree_estimator=partial(tree_estimator, num_rate_categories=num_rate_categories),
            initial_tree_estimator_rate_matrix_path=initial_tree_estimator_rate_matrix_path,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            use_cpp_counting_implementation=use_cpp_counting_implementation,
            optimizer_device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            edge_or_cherry=cherryml_type,
            cpp_counting_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_counting_command_line_suffix=cpp_counting_command_line_suffix,
            num_processes_tree_estimation=num_processes_tree_estimation,
            num_processes_counting=num_processes_counting,
            num_processes_optimization=num_processes_optimization,
            optimizer_initialization=optimizer_initialization,
            use_maximal_matching=use_maximal_matching,
            tree_dir=tree_dir,
        )

    # -------------------------------------------------------------------------
    # AA-3Di MODEL
    # -------------------------------------------------------------------------
    elif model_name == "AA-3Di":
        if msa_dir_3di is None:
            raise ValueError(
                "AA-3Di model requires msa_dir_3di to be provided (directory with 3Di MSAs)."
            )

        if tri_alphabet is None and tri_alphabet_path is None:
            logger.info(
                "No tri_alphabet provided: defaulting to standard amino-acid letters."
            )
            tri_alphabet = utils.get_amino_acids()

        if tri_alphabet is None and tri_alphabet_path is not None:
            with open(tri_alphabet_path, "r") as f:
                tri_alphabet = [line.strip() for line in f if line.strip()]

        outputs = aa3di_end_to_end_with_cherryml_optimizer(
            msa_dir_aa=msa_dir,
            msa_dir_3di=msa_dir_3di,
            families=families,
            tree_estimator=partial(tree_estimator, num_rate_categories=num_rate_categories),
            initial_tree_estimator_rate_matrix_path=initial_tree_estimator_rate_matrix_path,
            tri_alphabet=tri_alphabet,
            num_iterations=num_iterations,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            use_cpp_counting_implementation=use_cpp_counting_implementation,
            optimizer_device=optimizer_device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            edge_or_cherry=cherryml_type,
            cpp_counting_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_counting_command_line_suffix=cpp_counting_command_line_suffix,
            num_processes_tree_estimation=num_processes_tree_estimation,
            num_processes_counting=num_processes_counting,
            num_processes_optimization=num_processes_optimization,
            optimizer_initialization=optimizer_initialization,
            tree_dir=tree_dir,
            site_rates_dir=site_rates_dir,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # -------------------------------------------------------------------------
    # Write learned rate matrix
    # -------------------------------------------------------------------------
    learned_rate_matrix = read_rate_matrix(
        os.path.join(outputs["learned_rate_matrix_path"])
    )
    write_rate_matrix(
        learned_rate_matrix.to_numpy(),
        list(learned_rate_matrix.columns),
        output_path,
    )

    return output_path
