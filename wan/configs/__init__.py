import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B

WAN_CONFIGS = {
    'i2v-A14B': i2v_A14B,
}
