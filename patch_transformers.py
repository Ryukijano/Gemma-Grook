"""
Patch script to disable flash attention in Transformers
"""
import sys
import types

# Create mock modules to prevent actual imports
sys.modules['flash_attn'] = types.ModuleType('flash_attn')
sys.modules['flash_attn.flash_attn_interface'] = types.ModuleType('flash_attn.flash_attn_interface')
sys.modules['flash_attn.bert_padding'] = types.ModuleType('flash_attn.bert_padding')
sys.modules['flash_attn_2_cuda'] = types.ModuleType('flash_attn_2_cuda')

# Add required mock objects to prevent attribute errors
sys.modules['flash_attn'].flash_attn_interface = sys.modules['flash_attn.flash_attn_interface']
sys.modules['flash_attn.bert_padding'].index_first_axis = lambda *args, **kwargs: None
sys.modules['flash_attn.bert_padding'].pad_input = lambda *args, **kwargs: None
sys.modules['flash_attn.bert_padding'].unpad_input = lambda *args, **kwargs: None

# Continue with normal imports
print("Flash Attention successfully mocked. Continuing with imports...")

# Import the actual script - this will be replaced with execfile in the actual command
if __name__ == "__main__":
    print("Patch applied. Now proceed with normal execution.")
