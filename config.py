from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trl import ModelConfig
from trl.commands.cli_utils import SFTScriptArguments


@dataclass
class AriaModelConfig(ModelConfig):

    tokenizer_path: str = field(
        default=None,
        metadata={"help": "The path to the tokenizer."},
    )

    peft_model_path: str = field(
        default=None,
        metadata={"help": "The path to the PEFT model."}
    )


    freeze_projector: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the projector."}
    )

    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Wheather to freeze the LLM model."}
    )

    freeze_llm_player: List[int] = field(
        default=None,
        metadata={"help": "The indices of the LLM layers to freeze."}
    )

    moe_z_loss_coeff: float = field(
        default=1e-5,
        metadata={"help": "The coefficient for the z loss"}
    )

   

    moe_aux_loss_coeff: float = field(
        default= 900,
        metadata={
            "help": "The maximum size of the image after processing before being passed to the vision encoder.",
            "choices": [490, 980]
        }
    )


    def __post_init__(self):
        super().__post_init__()
        if self.max_image_size not in [490, 900]:
            raise ValueError("max_image_size must be either 490 or 980")




@dataclass
class AriaSFTScriptArguments(SFTScriptArguments):
    dataset_mixer: Optional[Dict[str, float]]  = field(
        default=None,
        metadata={
            "help": ("Datasets and their proportions to be used for training ift/rl.")
        }
    )