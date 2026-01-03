import pandas as pd
import json

model = "SSL_BAT_los_full"
dataset = "hirid"

# SSL_GRUD_m24_head_hirid
#file = "finetuning_results/pretrained_GRUD/hirid/head/summary_44a1def579.csv" # 100-3000
#file = "finetuning_results/pretrained_GRUD/hirid/head/summary_ea101927e8.csv" # 100-9506

# SSL_GRUD_m24_full_hirid
#file = "finetuning_results/pretrained_GRUD/hirid/full/summary_8bf747411d.csv" # 100-3000

# SSL_BAT_m24_head_hirid
#file = "finetuning_results/pretrained_BAT/hirid/head/summary_0184ca5db2.csv" # 100-3000
#file = "finetuning_results/pretrained_BAT/hirid/head/summary_bf2b627e24.csv" # 100-9506

# SSL_BAT_m24_full_hirid
#file = "finetuning_results/pretrained_BAT/hirid/full/summary_800ed55d7f.csv" # 100-3000
#file = "finetuning_results/pretrained_BAT/hirid/full/summary_449195ddc4.csv" # 100-9506

# SSL_BAT_los_full_hirid
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/full/summary_199ae04371.csv" # 100-3000
file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/full/summary_51cd92d9e1.csv" # other sizes 100-7000 were manually extracted due to interrupted run, 9000-9506

# SSL_BAT_los_head_hirid
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/head/summary_5da3a4022f.csv" # 2000
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/head/summary_b27d5720fc.csv" # 3000
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/head/summary_9612cdd520.csv" # 5000
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/head/summary_2650feac48.csv" # 7000
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/head/summary_b2599211f4.csv" # 9000
#file = "finetuning_results_regression/pretrained_BAT/LengthOfStay/hirid/head/summary_5d4fbd5396.csv" # 9506

# SSL_GRUD_los_head_hirid
#file = "finetuning_results_regression/pretrained_GRUD/LengthOfStay/hirid/head/summary_2b94025785.csv" # 100-3000
#file = "finetuning_results_regression/pretrained_GRUD/LengthOfStay/hirid/head/summary_fb5dad230b.csv" # 100-9506

# SSL_GRUD_los_full_hirid
#file = "finetuning_results_regression/pretrained_GRUD/LengthOfStay/hirid/full/summary_55b2046908.csv" # 100-3000
#file = "finetuning_results_regression/pretrained_GRUD/LengthOfStay/hirid/full/summary_4b0055d7c6.csv" # 100-9506

# SSL_GRUD_los_full_hirid
#file = "finetuning_results_regression/pretrained_GRUD/LengthOfStay/hirid/full/summary_55b2046908.csv" # 100-3000
#file = "finetuning_results_regression/pretrained_GRUD/LengthOfStay/hirid/full/summary_4b0055d7c6.csv" # 100-9506 

#file = "/work3/s185395/YAIB/finetuning_results/pretrained_BAT/eicu/full/summary_7d69adda7f.csv"
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
    if "los" in model:

        loss_mean = g.avg_test_loss.mean() 
        loss_sd   = g.avg_test_loss.std(ddof=1) 

        mae_mean = g.test_mae.mean() 
        mae_sd   = g.test_mae.std(ddof=1) 

        results[str(size)] = {
            "test/MAE":  f"{mae_mean:.4f} ± {mae_sd:.4f}",
            "test/loss": f"{loss_mean:.4f} ± {loss_sd:.4f}"
        }

    else:
        auc_mean = g.test_auroc.mean() * 100
        auc_sd   = g.test_auroc.std(ddof=1) * 100

        pr_mean = g.test_auprc.mean() * 100
        pr_sd   = g.test_auprc.std(ddof=1) * 100

        results[str(size)] = {
            "test/AUC":  f"{auc_mean:.2f} ± {auc_sd:.2f}",
            "test/PR":   f"{pr_mean:.2f} ± {pr_sd:.2f}"
        }

output = {
    "model": model,
    "dataset": dataset,
    "type": "subset",
    "sample_sizes": results
}

print(json.dumps(output, indent=2, ensure_ascii=False))

