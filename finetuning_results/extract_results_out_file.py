import re
import json

#model = "Transformer"
model = "BAT"

dataset = "eicu"
#filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25867256.out"# input log file, Transformer baseline, eicu
filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25898859.out"# input log file, BAT baseline, eicu

#dataset = "miiv"
#filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25866998.out" # input log file, Transformer baseline, miiv 
#filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25871756.out" # input log file, Transformer baseline, miiv samples 9506 rest 
#filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25895115.out" # input log file BAT baseline, miiv

#dataset = "mimic"
#filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25893016.out"# input log file, Transformer baseline, mimic 
#filepath = "/work3/s185395/YAIB/hpc_output/SL_output_25893011.out"# input log file, BAT baseline, mimic 

output_path = f"/work3/s185395/YAIB/finetuning_results/{model}/{dataset}/baseline_subsets_rest.json"  # where to save the results

def parse_log_file(filepath):
    results = []
    with open(filepath, "r") as f:
        content = f.read()

    # Split into chunks for each run
    chunks = content.split("🔍 Subsetting training data")

    for chunk in chunks:
        if not chunk.strip():
            continue

        # Extract number of samples and seed
        match_seed = re.search(r"to (\d+) samples \(seed=(\d+)\)", chunk)
        if not match_seed:
            continue
        num_samples = int(match_seed.group(1))
        seed = int(match_seed.group(2))

        # Extract metrics
        match_auc = re.search(r"test/AUC\s*│\s*([\d.eE+-]+)", chunk)
        match_pr = re.search(r"test/PR\s*│\s*([\d.eE+-]+)", chunk)
        match_loss = re.search(r"test/loss\s*│\s*([\d.eE+-]+)", chunk)

        if match_auc and match_pr and match_loss:
            result = {
                "num_samples": num_samples,
                "seed": seed,
                "test/AUC": float(match_auc.group(1)),
                "test/PR": float(match_pr.group(1)),
                "test/loss": float(match_loss.group(1))
            }
            results.append(result)

    return results

# Parse + save
parsed_results = parse_log_file(filepath)

with open(output_path, "w") as f:
    json.dump(parsed_results, f, indent=2)

print(f"✅ Extracted {len(parsed_results)} results and saved to {output_path}")

