import json
import os
import re
from pathlib import Path

def get_next_prompt_idx(data: dict) -> int:
    indices = []
    for key in data.keys():
        match = re.match(r'prompt(\d+)', key)
        if match:
            indices.append(int(match.group(1)))
    
    return max(indices)+1 if indices else 0
    
def save_prompt(prompt_text_txt: Path, pe_json_path: Path) -> None:
    with open(prompt_text_txt, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    
    with open(pe_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    idx = get_next_prompt_idx(data)
    new_prompt =f'prompt{idx}'
    new_entry = {
        'comment': '',
        'prompt_text': prompt_text
    }
    
    data[new_prompt] = new_entry

    with open(pe_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    prompt_text_txt = './dataset/prompt_text.txt'
    pe_json_path = './dataset/pe.json'
        
    save_prompt(prompt_text_txt, pe_json_path)
