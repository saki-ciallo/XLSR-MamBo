# train_config.py

from dataclasses import dataclass
from dataset_load import asv19_dataloader
import model_build as mb 


@dataclass
class TaskConfig:
    # task_name: str
    model_classifier: type
    model_config: dict
    # dataloader_train: callable
    # dataloader_dev: callable
    

MODEL_CONFIG = mb.Mamba2Config(d_model=128, d_state=64, d_conv=4, expand=2, headdim=32,
                                mlp_type="SwiGLU",
                                mha_num_heads=4,
                                mha_mlp_dim=0,
                                mha_d_conv=4,
                                gdn_num_heads=4,
                                gdn_head_dim=32,
                                residual_in_fp32=True,
                                fused_add_norm=True,
                                use_mem_eff_path=True, # Hydra 使用该参数
                                )

# 手动配置使用什么模型
TASK_CONFIGS = {
    "xlsr_Hydra": TaskConfig(
    model_classifier=mb.F4_XLSR300M_General,
    model_config={
        "mamba_config": MODEL_CONFIG.replace(d_conv=7), # hydra default d_conv=7
        },
    ),
    "xlsr_Mamba2": TaskConfig(
    model_classifier=mb.F4_XLSR300M_General,
    model_config={
        "mamba_config": MODEL_CONFIG,
        },
    ),
    "xlsr_Mamba1": TaskConfig(
    model_classifier=mb.F4_XLSR300M_General,
    model_config={
        "mamba_config": MODEL_CONFIG.replace(d_state=16),
        },
    ),
    "xlsr_GDN": TaskConfig(
    model_classifier=mb.F4_XLSR300M_General,
    model_config={
        "mamba_config": MODEL_CONFIG,
        },
    ),
    
}