import json
import jsonlines

# Load MITRE ATT&CK dataset
with open("enterprise-attack.json") as f:
    data = json.load(f)

# Create instruction-style prompt-completion pairs
dataset = []
for obj in data["objects"]:
    if obj.get("type") == "attack-pattern":
        external_id = obj['external_references'][0]['external_id']
        prompt = f"Explain {external_id} in plain language."
        completion = obj.get("description", "").split("\n")[0]
        dataset.append({"prompt": prompt, "completion": completion})

# Save as .jsonl for training
with jsonlines.open("mitre_prompts.jsonl", mode='w') as writer:
    for entry in dataset:
        writer.write(entry)

print("✅ MITRE prompts exported to mitre_prompts.jsonl")
