"""RL-QLS implementation package, including multi-molecule GNN control."""
from .env import RLQLSEnv
from .model import BranchModel, BranchResult, boltzmann_population
from .materials import build_cah16_surrogate, build_h3o130_surrogate
from .dqn import DQNConfig, QNetwork, train_dqn
from .multitask import (
    MultiMoleculeRLQLSEnv,
    MoleculeTask,
    MoleculeTaskRegistry,
    build_default_registry,
)
from .gnn import ChemistryConditionedQNetwork, GNNQConfig
from .multitask.trainer import MultiTaskDQNConfig, train_multitask_dqn

__all__ = [
    "RLQLSEnv",
    "BranchModel",
    "BranchResult",
    "boltzmann_population",
    "build_cah16_surrogate",
    "build_h3o130_surrogate",
    "DQNConfig",
    "QNetwork",
    "train_dqn",
    "ChemistryConditionedQNetwork",
    "GNNQConfig",
    "MultiMoleculeRLQLSEnv",
    "MultiTaskDQNConfig",
    "MoleculeTask",
    "MoleculeTaskRegistry",
    "build_default_registry",
    "train_multitask_dqn",
]
