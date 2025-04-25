from datasets import load_dataset

dataset = load_dataset("VatsaDev/worldbuild", split="train")
dataset.to_json("datasets/dataset.jsonl", orient="records", lines=True)