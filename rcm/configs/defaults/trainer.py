from hydra.core.config_store import ConfigStore
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import PLACEHOLDER, LazyDict
from imaginaire.trainer import ImaginaireTrainer
from rcm.trainers.trainer_distillation import ImaginaireTrainer_Distill

TRAINER: LazyDict = L(ImaginaireTrainer)(config=PLACEHOLDER)
TRAINER_DISTILL: LazyDict = L(ImaginaireTrainer_Distill)(config=PLACEHOLDER)


def register_trainer():
    cs = ConfigStore.instance()
    cs.store(group="trainer", package="trainer.type", name="standard", node=TRAINER)
    cs.store(group="trainer", package="trainer.type", name="distill", node=TRAINER_DISTILL)
