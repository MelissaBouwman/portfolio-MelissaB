from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# HIER heb ik jouw mapnaam ingevuld:
experiment_path = str(Path("/Users/melissabouwman/Documents/GitHub/portfolio-MelissaB/logs/ray/train_func_2026-01-14_21-32-11").resolve())

def plot_results():
    # Check of het pad bestaat
    if not Path(experiment_path).exists():
        print(f"LET OP: Kan map {experiment_path} niet vinden. Check de naam!")
        return

    print(f"Laden van {experiment_path}...")
    analysis = ExperimentAnalysis(experiment_path)
    df = analysis.dataframe()

    # Even printen wat we hebben gevonden
    print(f"Aantal runs gevonden: {len(df)}")
    print("Kolommen:", df.columns)
    
    plt.figure(figsize=(12, 7))
    
    # De Scatterplot
    sns.scatterplot(
        data=df,
        x="config/lr",          # Learning Rate op X
        y="accuracy",           # Accuracy op Y (hoger is beter)
        hue="config/use_batchnorm", # Kleurtje voor wel/geen Batchnorm
        style="config/use_batchnorm",
        s=150,                  # Dikke punten
        palette="viridis"
    )

    plt.xscale("log") # Belangrijk voor learning rate!
    plt.title("Hypertuning: Effect van Batch Norm bij hoge Learning Rate")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Opslaan
    save_path = "resultaat_grafiek.png"
    plt.savefig(save_path)
    print(f"Succes! Grafiek is opgeslagen als {save_path}")
    # plt.show() # Zet dit aan als je geen plaatje ziet verschijnen

if __name__ == "__main__":
    plot_results()