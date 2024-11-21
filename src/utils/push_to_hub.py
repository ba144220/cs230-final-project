import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoModelForCausalLM

from dotenv import load_dotenv

load_dotenv()

@dataclass
class PushToHubArguments:
    model_name_or_path: str = field()
    adapter_path: str = field()
    hf_organization: str = field()
    repo_id: str = field()

if __name__ == "__main__":
    parser = HfArgumentParser(PushToHubArguments)
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.load_adapter(args.adapter_path)
    
    model.push_to_hub(f"{args.hf_organization}/{args.repo_id}", token=os.getenv("HF_TOKEN"))
    
