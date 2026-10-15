"""Local Qwen2.5 generation for the RAG approach.

Weights are downloaded from Hugging Face on first use; generation then runs
locally with no hosted API.
"""
import os


class LocalGenerator:
    def __init__(self, model_name=None, tokenizer=None, model=None):
        if tokenizer is not None and model is not None:
            self.tokenizer, self.model = tokenizer, model
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Install local generation dependencies: pip install -e ".[rag,local,test]"') from exc
        self.torch = torch
        # 0.5B on CPU, 1.5B when a GPU is available.
        model_name = model_name or os.getenv('LOCAL_RAG_MODEL') or (
            'Qwen/Qwen2.5-1.5B-Instruct' if torch.cuda.is_available()
            else 'Qwen/Qwen2.5-0.5B-Instruct'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def chat(self, messages, max_new_tokens=220):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        tokens = output[0][inputs['input_ids'].shape[-1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
