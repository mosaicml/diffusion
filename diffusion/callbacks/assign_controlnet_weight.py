from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from composer import Callback, Logger, State
from composer.core import get_precision_context
from torch.nn.parallel import DistributedDataParallel
from diffusers import ControlNetModel, UNet2DConditionModel

class AssignControlNet(Callback):
    """Assigns Controlnet weights to the controlnet from the Unet after composer loads the checkpoint

    Args:
        use_fsdp: whether or not the model is FSDP wrapped
    """

    def __init__(self, use_fsdp):
        self.use_fsdp = use_fsdp

    def process_controlnet(self, controlnet: ControlNetModel, unet: UNet2DConditionModel):
        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
        controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        if controlnet.class_embedding:
            controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

        if hasattr(controlnet, "add_embedding"):
            controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
    
    def after_load(self, state: State, logger: Logger):
        # Get the model object if it has been wrapped by DDP to access the image generation function.
        if isinstance(state.model, DistributedDataParallel):
            model = state.model.module
        else:
            model = state.model

        # Load checkpoint
        if model.load_controlnet_from_composer:
            with get_precision_context(state.precision):
                if self.use_fsdp:
                    with FSDP.summon_full_params(model.unet, recurse = True, writeback = False):
                        with FSDP.summon_full_params(model.controlnet, recurse = True, writeback = True):
                            self.process_controlnet(model.controlnet, model.unet)
                
                else:
                    self.process_controlnet(model.controlnet, model.unet)
            