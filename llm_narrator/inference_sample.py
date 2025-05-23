from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_path = "./llama2_mitre"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe("Explain T1059.003 in plain language.", max_length=60)[0]["generated_text"]
print("🧠 LLM Output:\n", result)
