import torch
import json
import numpy as np
import os
import argparse
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModel
)
import re
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from image_transformation_func import ImageTransformer
from qwen_vl_utils import process_vision_info
from PIL import Image as PILImage
import cv2
from PIL.Image import Image as PILImageType
import gc
from collections import OrderedDict
from utils.internVL import load_image_internVL

_original_fromarray = PILImage.fromarray
def _safe_fromarray(obj, *args, **kwargs):
    if isinstance(obj, PILImageType):
        return obj
    return _original_fromarray(obj, *args, **kwargs)
PILImage.fromarray = _safe_fromarray


YES_PROMPT_TEMPLATE = """
You are given a yes/no question, where the answer is **yes**. 
You should convert the question and the response to a single factual statement. As a result, the factual statement you generate should be a positive sentence.

Your response must include **only** the factual statement, in the following format:

Factual Statement: <factual statement>  

Example  
Input:  
Question: Is the young boy in the image happy?

Output:  
Factual Statement: The young boy in the image is happy.  

Now complete the task:  
Question: {question}
"""

NO_PROMPT_TEMPLATE = """
You are given a yes/no question, where the answer is **no**.
You should convert the question and the response to a single factual statement. As a result, the factual statement you generate should be a negative sentence.

Your response must include **only** the factual statement, in the following format:

Factual Statement: <factual statement>  

Example  
Input:  
Question: Is the young boy in the image happy?

Output:  
Factual Statement: The young boy in the image is not happy.  

Now complete the task:  
Question: {question}
"""

TASK_SPECS = OrderedDict({
    "lexical_substitution": (
        "Lexical substitution",
        "Replace words/short phrases with synonyms or near-synonyms while preserving exact meaning. "
    ),
    "syntactic_restructuring": (
        "Syntactic restructuring",
        "Rewrite the sentence using a different syntactic structure (fronting, inversion, clefting) while preserving meaning. "
    ),
    "phrasal_rephrasing": (
        "Phrasal rephrasing",
        "Replace multi-word phrases or constructions with equivalent phrasings (verb phrase / nominalization / prepositional alternatives). "
    ),
    "add_redundant_neutral_modifiers": (
        "Add redundant/neutral modifiers",
        "Produce a paraphrase by adding neutral modifiers (e.g., \"clearly\", \"visible\") that do not change truth conditions. Avoid qualifiers that introduce uncertainty. "
    ),
})

PHI_2_MODEL_ID = "microsoft/phi-2"

IMAGE_TRANSFORMATIONS = [
    'original',
    "gaussian_noise",
    "impulse_noise",
    "gridmask",
    "elastic_transform",
    "salt_and_pepper_noise",
    "style_augmentation",
    "poisson_noise",
    "nl_means",
    "motion_blur",
    "speckle_noise",
]

def to_binary(answer_text: str) -> int:
        if not answer_text:
            return 0
        t = answer_text.strip().lower()
        if "assistant:" in t:
            t = t.split("assistant:", 1)[1].strip()
        match = re.search(r"\b(true|false|yes|no)\b", t)
        if match:
            token = match.group(1)
            return 1 if token in ("true", "yes") else 0
        if ("false" in t) or ("no" in t):
            return 0
        if ("true" in t) or ("yes" in t):
            return 1
        return 0

def make_task_prompt(task_name: str, instruction: str, factual: str) -> str:
    factual = (factual or "").strip()
    return (
        "You are a precise paraphrasing assistant. For each input, produce a single paraphrase that preserves the "
        "original meaning exactly. Do not add new facts or omit information."
        f"Task: {task_name}\n"
        f'Input: "{factual}"\n'
        f"Instruction: {instruction}"
        "\nReturn only the paraphrase and do not add anything furthermore.\n\n"
        "Paraphrase:"
    )

def run_phi2_inference(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in text:
        text = text.replace(prompt, "").strip()
    return text

def generate_factual_statement(question, model, tokenizer):
    results = {}

    yes_prompt = YES_PROMPT_TEMPLATE.format(question=question)
    yes_raw = run_phi2_inference(model, tokenizer, yes_prompt, max_new_tokens=50)
    affirmative_match = re.search(r"Factual Statement:\s*(.+)", yes_raw)
    affirmative = affirmative_match.group(1).strip() if affirmative_match else yes_raw.strip()
    affirmative = affirmative.strip('"').strip("'")
    results["affirmative"] = affirmative

    no_prompt = NO_PROMPT_TEMPLATE.format(question=question)
    no_raw = run_phi2_inference(model, tokenizer, no_prompt, max_new_tokens=50)
    contradictory_match = re.search(r"Factual Statement:\s*(.+)", no_raw)
    contradictory = contradictory_match.group(1).strip() if contradictory_match else no_raw.strip()
    contradictory = contradictory.strip('"').strip("'")
    results["contradictory"] = contradictory

    for polarity in ["affirmative", "contradictory"]:
        base_text = results[polarity]
        for key, (task_name, instruction) in TASK_SPECS.items():
            task_prompt = make_task_prompt(task_name, instruction, base_text)
            variation = run_phi2_inference(model, tokenizer, task_prompt, max_new_tokens=100)
            variation = variation.split("\n")[0].strip().strip('"')
            results[f"{polarity}_{key}"] = variation

    return results

def llava_question(question):
    Prompt_prefix = """USER:<image>\n<"""
    prompt = Prompt_prefix + question.strip() + "ASSISTANT:"
    return prompt
    
def blip_question(question):
    prompt = question.strip()
    return prompt

def qwen_question(question):
    prompt = question.strip()
    return prompt

def internvl_question(question):
    Prompt_prefix = """Answer to the question with Yes or No.
    """
    prompt = Prompt_prefix + question.strip()
    return prompt

def llava_prompt(factual_claim):
    Prompt_prefix = """USER:<image>
    is this claim correct about the image?
    claim: """
    Prompt_suffix = """
    print \nYes\n if the claim is correct and print \nNo\n if the claim is not correct
    """
    cleaned_fc = preprocess_factualclaim(factual_claim)
    prompt = Prompt_prefix + cleaned_fc.strip() + Prompt_suffix + "\nASSISTANT:"
    return prompt
    
def blip_prompt_img(factual_claim):
    Prompt_prefix = """Check the Image and following Claim then answer with True or False.
            Claim: """
    Prompt_suffix = """
            Is the Claim true or false?
            """
    cleaned_fc = preprocess_factualclaim(factual_claim)
    prompt = Prompt_prefix + cleaned_fc.strip() + Prompt_suffix
    return prompt

def blip_prompt_fs(factual_claim):
    Prompt_prefix = """Check the Image and following Claim then answer with True or False.
    Claim: """
    Prompt_suffix = """
    Is the Claim true or false?
    """
    cleaned_fc = preprocess_factualclaim(factual_claim)
    prompt = Prompt_prefix + cleaned_fc.strip() + Prompt_suffix
    return prompt

def qwen_prompt_img(factual_claim):
    return blip_prompt_img(factual_claim)

def qwen_prompt_fs(factual_claim):
    return blip_prompt_fs(factual_claim)

def internvl_prompt_img(factual_claim):
    Prompt_prefix = """Direct evaluate Claim whether True or False only. Do not explain step by step. Do not include analysis.
    Claim: """
    Prompt_suffix = """
    Is the Claim true or false?
    """
    cleaned_fc = preprocess_factualclaim(factual_claim)
    prompt = Prompt_prefix + cleaned_fc.strip() + Prompt_suffix
    return prompt

def internvl_prompt_fs(factual_claim):
    Prompt_prefix = """Direct evaluate Claim only. Do not explain step by step. Do not include analysis.
    Claim: """
    Prompt_suffix = """
    Is the Claim true or false?
    """
    cleaned_fc = preprocess_factualclaim(factual_claim)
    prompt = Prompt_prefix + cleaned_fc.strip() + Prompt_suffix
    return prompt

def preprocess_factualclaim(factual_claim):
    # Remove double quotes only
    cleaned = factual_claim.replace('"', '')
    # Normalize spaces and strip leading/trailing whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned

def process_baseline(image_path, question, model_name, model, processor, device):
    image_transformer = ImageTransformer(image_path=image_path)
    image_org = image_transformer.original_image
    
    if model_name == "llava":
        prompt = llava_question(question) 
    elif model_name == "blip":
        prompt = blip_question(question)
    elif model_name == "qwen":
        prompt = qwen_question(question)
    elif model_name == "internvl":
        prompt = internvl_question(question)
    else:
        raise ValueError(f"Model {model_name} not supported")

    generated_text = ""
    
    if model_name == "llava":
        with torch.no_grad():
            inputs = processor(images=image_org, text=prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
    elif model_name == "blip":
        with torch.no_grad():
            inputs = processor(images=image_org, text=prompt, return_tensors="pt").to(device)
            input_ids_length = inputs['input_ids'].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=30)
            generated_ids = outputs[:, input_ids_length:]
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
    elif model_name == "qwen":
         with torch.no_grad():
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_org}, 
                    {"type": "text", "text": prompt}
                ]
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

    elif model_name == "internvl":
        with torch.no_grad():
            pixel_values = load_image_internVL(image_org, max_num=4).to(torch.bfloat16).to(device)

            generation_config = dict(
                num_beams=1,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=processor.eos_token_id
            )
            response_text = model.chat(
                processor, 
                pixel_values, 
                prompt, 
                generation_config
            )
            generated_text = response_text.strip()
            # print(prompt)
            # print(generated_text)

    # print(generated_text)
    return to_binary(generated_text)

def process_image(image_path, factual_claim, model_name, model, processor, device):
    affirmative_factual_claim = factual_claim["affirmative"]
    contradictory_factual_claim = factual_claim["contradictory"]

    if model_name == "llava":
        affirmative_prompt = llava_prompt(affirmative_factual_claim)
        contradictory_prompt = llava_prompt(contradictory_factual_claim)
    elif model_name == "blip":
        affirmative_prompt = blip_prompt_img(affirmative_factual_claim)
        contradictory_prompt = blip_prompt_img(contradictory_factual_claim)
    elif model_name == "qwen":
        affirmative_prompt = qwen_prompt_img(affirmative_factual_claim)
        contradictory_prompt = qwen_prompt_img(contradictory_factual_claim)
    elif model_name == "internvl":
        affirmative_prompt = internvl_prompt_img(affirmative_factual_claim)
        contradictory_prompt = internvl_prompt_img(contradictory_factual_claim)
    else:
        raise ValueError(f"Model {model_name} not supported") 

    image_transformer = ImageTransformer(image_path=image_path)

    final_results = {
        "affirmative": {},
        "contradictory": {}
    }

    for method_name in tqdm(IMAGE_TRANSFORMATIONS, desc="Processing Image Transformations"):
        if method_name == "original":
            transformed_image = image_transformer.original_image
        else:
            method = getattr(image_transformer, method_name)
            transformed_image = method()

        claims_to_process = [
            ("affirmative", affirmative_prompt),
            ("contradictory", contradictory_prompt)
        ]

        for claim_type, current_prompt in claims_to_process:
            generated_text = ""

            if model_name == "llava":
                with torch.no_grad():
                    inputs = processor(images=transformed_image, text=current_prompt, return_tensors="pt").to(device)
                    generated_ids = model.generate(**inputs, max_new_tokens=500)
                    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            elif model_name == "blip":
                with torch.no_grad():
                    inputs = processor(images=transformed_image, text=current_prompt, return_tensors="pt").to(device)
                    input_ids_length = inputs['input_ids'].shape[1]
                    outputs = model.generate(
                            **inputs,
                            max_new_tokens=30,
                    )
                    generated_ids = outputs[:, input_ids_length:]
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            elif model_name == "qwen":
                with torch.no_grad():
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": transformed_image,
                                },
                                {"type": "text", "text": current_prompt},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    generated_ids = model.generate(**inputs, max_new_tokens=30)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    generated_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
            
            elif model_name == "internvl":
                with torch.no_grad():
                    pixel_values = load_image_internVL(transformed_image, max_num=4).to(torch.bfloat16).to(device)
                    
                    generation_config = dict(
                        num_beams=1,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=processor.eos_token_id
                    )
                    response_text = model.chat(
                        processor, 
                        pixel_values, 
                        current_prompt, 
                        generation_config
                    )
                    generated_text = response_text.strip()
                    # print(current_prompt)
                    # print(generated_text)
                    
            # print(generated_text)
            final_results[claim_type][method_name] = to_binary(generated_text)

    return final_results

def process_factual_statement(image_path, factual_statement, model_name, model, processor, device):
    TEXT_TRANSFORMATIONS = [
        "lexical_substitution",
        "syntactic_restructuring",
        "phrasal_rephrasing",
        "add_redundant_neutral_modifiers",
    ]

    image_transformer = ImageTransformer(image_path=image_path)
    image_org = image_transformer.original_image

    final_results = {
        "affirmative": {},
        "contradictory": {}
    }

    for claim_type in ["affirmative", "contradictory"]:
        
        for transform_name in tqdm(TEXT_TRANSFORMATIONS, desc=f"Processing {claim_type} text variations"):

            dict_key = f"{claim_type}_{transform_name}"

            if dict_key in factual_statement:
                current_claim = factual_statement[dict_key]
            else:
                current_claim = factual_statement[claim_type]

            if model_name == "llava":
                prompt = llava_prompt(current_claim)
            elif model_name == "blip":
                prompt = blip_prompt_fs(current_claim)
            elif model_name == "qwen":
                prompt = qwen_prompt_fs(current_claim)
            elif model_name == "internvl":
                prompt = internvl_prompt_fs(current_claim)
            else:
                raise ValueError(f"Model {model_name} not supported")

            generated_text = ""
            
            if model_name == "llava":
                with torch.no_grad():
                    inputs = processor(images=image_org, text=prompt, return_tensors="pt").to(device)
                    generated_ids = model.generate(**inputs, max_new_tokens=300)
                    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            elif model_name == "blip":
                with torch.no_grad():
                    inputs = processor(images=image_org, text=prompt, return_tensors="pt").to(device)
                    input_ids_length = inputs['input_ids'].shape[1]
                    outputs = model.generate(**inputs, max_new_tokens=30)
                    generated_ids = outputs[:, input_ids_length:]
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            elif model_name == "qwen":
                with torch.no_grad():
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_org},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    generated_ids = model.generate(**inputs, max_new_tokens=30)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    generated_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

            elif model_name == "internvl":
                with torch.no_grad():
                    pixel_values = load_image_internVL(image_org, max_num=4).to(torch.bfloat16).to(device)
                    
                    generation_config = dict(
                        num_beams=1,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=processor.eos_token_id
                    )
                    response_text = model.chat(
                        processor, 
                        pixel_values, 
                        prompt, 
                        generation_config
                    )
                    generated_text = response_text.strip()
                    # print(prompt)
                    # print(generated_text)
                    
            # print(generated_text)
            final_results[claim_type][transform_name] = to_binary(generated_text)

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the JSON file containing samples.")
    parser.add_argument("--model-name", type=str, default="llava", choices=["llava", "blip", "qwen", "internvl"],
                        help="VLM model to use.")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results.")
    parser.add_argument("--step", type=int, default=4, choices=[1, 2, 3, 4],
                        help="1: Phi-2 Text Gen only. 2: Add Baseline. 3: Add Image Perturbations. 4: Add Text Perturbations (Full).")
    
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples from {args.input_file}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.step == 1: # step 1
        final_results_path = os.path.join(args.output_dir, f"results_full_{os.path.basename(os.path.dirname(args.input_file))}.json")
    else: # step 2, 3, 4
        if 'results_full_' not in  os.path.splitext(os.path.basename(args.input_file))[0]: # step 1 is not done
            final_results_path = os.path.join(args.output_dir, f"results_full_{os.path.basename(os.path.dirname(args.input_file))}_{args.model_name}.json")
        else: # step 1 was done
            if args.model_name not in os.path.splitext(os.path.basename(args.input_file))[0]: # step 2 is not done
                final_results_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input_file))[0]}_{args.model_name}.json") # step 2
            else: # step 2 was done
                final_results_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.input_file))[0]}.json") # step 3 or higher
                
        

    print("\n--- [Step 1] Loading Phi-2 for Batch Text Generation ---")
    needs_phi2 = any('factual_statement' not in item for item in dataset)

    if needs_phi2:
        tokenizer = AutoTokenizer.from_pretrained(PHI_2_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        phi_model = AutoModelForCausalLM.from_pretrained(
            PHI_2_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        print("Generating factual statements for all samples...")
        for item in tqdm(dataset, desc="Phi-2 Inference"):
            try:
                if 'factual_statement' not in item:
                    facts = generate_factual_statement(item['query'], phi_model, tokenizer)
                    item['factual_statement'] = facts
            except Exception as e:
                print(f"Error generating facts for ID {item.get('id', 'unknown')}: {e}")
        
        print("--- Unloading Phi-2 to free memory ---")
        del phi_model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("Factual statements found in input. Skipping Phi-2 generation.")

    processed_samples = dataset 
    with open(final_results_path, "w") as f:
        json.dump(processed_samples, f, indent=4)

    if args.step == 1:
        print(f"\n[INFO] Step 1 complete. Stopping execution.")
        exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Initializing {args.model_name} on {device} for Steps 2-{args.step} ---")

    model = None
    processor = None

    if args.model_name == "llava":
        model_path = "llava-hf/llava-1.5-7b-hf"
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(device).eval()
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        except Exception as e:
            print(f"Error loading Llava: {e}")
            exit(1)

    elif args.model_name == "blip":
        #model_path = "Salesforce/instructblip-vicuna-7b"
        local_model_path = "/home/user01/.cache/huggingface/hub/models--Salesforce--instructblip-vicuna-7b/snapshots/19103d0c5b5263c8a7891012e08573439fb6607f"
        try:
            #model = InstructBlipForConditionalGeneration.from_pretrained(
            #    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            #).to(device).eval()
            #processor = InstructBlipProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = InstructBlipForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
            processor = InstructBlipProcessor.from_pretrained(local_model_path)
        except Exception as e:
            print(f"Error loading Blip: {e}")
            exit(1)

    elif args.model_name == "qwen":
        # model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        local_model_path = "/home/user01/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
        try:
            # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #     model_path, torch_dtype=torch.float16, device_map="auto"
            # ).eval()
            # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_model_path, torch_dtype=torch.float16, device_map="auto"
            ).eval()
            processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading Qwen: {e}")
            exit(1)

    elif args.model_name == "internvl":
        # model_path = "OpenGVLab/InternVL3_5-4B"
        local_model_path = "/home/aezzati/.cache/huggingface/hub/models--OpenGVLab--InternVL3_5-4B/snapshots/481f6e32467eab4e922ccd7fd6cf420441a62331"
        try:
            # model = AutoModel.from_pretrained(
            #     model_path,
            #     torch_dtype=torch.bfloat16, 
            #     trust_remote_code=True
            # ).to(device).eval()
            # processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                local_model_path,
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(device).eval()
            processor = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True) 
        except Exception as e:
            print(f"Error loading InternVL: {e}")
            exit(1)

    print(f"\n--- Running Pipeline up to Step {args.step} on {len(processed_samples)} samples ---")

    for i, item in enumerate(tqdm(processed_samples, desc="Total Progress")):
        if i>= 5000:
            break
        sample_id = item['id']
        question = item['query']
        raw_path = item['image']
        image_path = f"../{raw_path}"
        factual_statement = item.get('factual_statement')

        if not os.path.exists(image_path):
            print(f"Warning: Image not found for ID {sample_id} at {image_path}. Skipping.")
            continue

        if "results" not in processed_samples[i]:
            processed_samples[i]["results"] = {
                "meta": {
                    "model": args.model_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

        try:
            if args.step >= 2:
                if "baseline_original" not in processed_samples[i]["results"]:
                    baseline_result = process_baseline(
                        image_path=image_path,
                        question=question,
                        model_name=args.model_name,
                        model=model, processor=processor, device=device
                    )
                    processed_samples[i]["results"]["baseline_original"] = baseline_result

            if args.step >= 3:
                if "image_perturbations" not in processed_samples[i]["results"]:
                    image_results = process_image(
                        image_path=image_path, 
                        factual_claim=factual_statement, 
                        model_name=args.model_name,
                        model=model,        
                        processor=processor, 
                        device=device       
                    )
                    processed_samples[i]["results"]["image_perturbations"] = image_results

            if args.step >= 4:
                if "text_perturbations" not in processed_samples[i]["results"]:
                    text_results = process_factual_statement(
                        image_path=image_path, 
                        factual_statement=factual_statement, 
                        model_name=args.model_name,
                        model=model,        
                        processor=processor, 
                        device=device       
                    )
                    processed_samples[i]["results"]["text_perturbations"] = text_results

            with open(final_results_path, "w") as f:
                json.dump(processed_samples, f, indent=4)
        
        except Exception as e:
            print(f"Error processing ID {sample_id}: {e}")
            continue

    print(f"\nSuccess! Results for Step {args.step} saved to: {final_results_path}")
