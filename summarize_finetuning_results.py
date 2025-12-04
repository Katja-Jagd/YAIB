import pandas as pd
import json

model = "SSL_BAT_full"
dataset = "eicu"


file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/eicu/full/summary_7d69adda7f.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/full/summary_bc27b0d2bc.csv"

#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/miiv/full/summary_4e3f34555e.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/miiv/full/summary_24631220ba.csv" 

# mimic 9506 head 0.01
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_342e11cf87.csv"

# mimic 9506 head 3.2e-3
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_0cc0edceb0.csv"


#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_b98d59719c.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/full/summary_bc27b0d2bc.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_f4b02911c2.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/miiv/head/summary_a6e1c2ccdc.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/miiv/full/summary_4e3f34555e.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/full/summary_362c7c1b2c.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_33fc13c06c.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_e3ce2dacc5.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/eicu/head/summary_a26e03d739.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_0cc0edceb0.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_33fc13c06c.csv"
#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/mimic/head/summary_430f02aeda.csv"

df = pd.read_csv(file)

print(df.lr)
results = {}

for size, g in df.groupby("size"):
    auc_mean = g.test_auroc.mean() * 100
    auc_sd   = g.test_auroc.std(ddof=1) * 100

    pr_mean = g.test_auprc.mean() * 100
    pr_sd   = g.test_auprc.std(ddof=1) * 100

    loss_mean = g.avg_test_loss.mean() * 100
    loss_sd   = g.avg_test_loss.std(ddof=1) * 100

    results[str(size)] = {
        "test/AUC":  f"{auc_mean:.2f} ± {auc_sd:.2f}",
        "test/PR":   f"{pr_mean:.2f} ± {pr_sd:.2f}",
        "test/loss": f"{loss_mean:.2f} ± {loss_sd:.2f}"
    }

output = {
    "model": model,
    "dataset": dataset,
    "type": "subset",
    "sample_sizes": results
}

print(json.dumps(output, indent=2, ensure_ascii=False))

