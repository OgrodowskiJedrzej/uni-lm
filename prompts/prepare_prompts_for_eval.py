import yaml
import os


def export_prompts():
    with open("backend/prompts.yaml", "r") as f:
        data = yaml.safe_load(f)

    os.makedirs("prompts/.prompts_cache", exist_ok=True)

    for agent_name, config in data["agents"].items():
        with open(f"prompts/.prompts_cache/{agent_name}.txt", "w") as f:
            f.write(config["system_prompt"])


if __name__ == "__main__":
    export_prompts()
