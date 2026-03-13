import torch
from unsloth import FastLanguageModel
import os

class ChatBot:
    def __init__(self, model_dir: str = "models/qwen_pdf_model"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self):
        """Load fine-tuned model"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model not found at {self.model_dir}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        FastLanguageModel.for_inference(self.model)
        self.loaded = True
    
    def chat(self, message: str, max_length: int = 512) -> str:
        """Generate response to user message"""
        if not self.loaded:
            self.load_model()
        
        prompt = f"### Instruction:\n{message}\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
