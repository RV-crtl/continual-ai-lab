"""Optional local model using promoted PEFT weights; isolated from Ollama install."""
from __future__ import annotations
from pathlib import Path
from .training_registry import read_active


class HuggingFaceClient:
    def __init__(self, base_model: str = 'Qwen/Qwen3-1.7B', adapter: Path | None = None,
                 use_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        self.model = base_model + (' + promoted adapter' if adapter else ' (untrained)')
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not torch.cuda.is_available():
            raise RuntimeError('Hugging Face backend requires an NVIDIA GPU with CUDA')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        options = {'device_map': {'': 0}, 'torch_dtype': dtype, 'trust_remote_code': False}
        if use_4bit:
            options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True)
        self.net = AutoModelForCausalLM.from_pretrained(base_model, **options)
        if adapter:
            self.net = PeftModel.from_pretrained(self.net, str(adapter), is_trainable=False)
        self.net.eval()

    @classmethod
    def active(cls, root: Path = Path('data/local/training')):
        info = read_active(root)
        if not info:
            raise RuntimeError('No promoted adapter. Run auto-training and pass both benchmark gates first.')
        return cls(info['base_model'], Path(info['adapter']))

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 192) -> str:
        messages = [{'role':'system','content':'Be accurate, concise and follow the requested output format.'},
                    {'role':'user','content':prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        inputs = {k:v.to('cuda') for k,v in inputs.items()}
        params = {'max_new_tokens':max_tokens, 'pad_token_id':self.tokenizer.pad_token_id,
                  'do_sample':temperature>0}
        if temperature>0: params['temperature']=temperature
        with self.torch.inference_mode():
            out = self.net.generate(**inputs, **params)
        return self.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
