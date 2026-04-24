from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    path = Path("reports/grid_search_results.csv")
    if not path.exists():
        raise FileNotFoundError("Run grid_search.py first.")

    df = pd.read_csv(path)
    pivot = df.pivot_table(index="ema_period", columns="rr_target", values="score", aggfunc="mean")

    plt.figure(figsize=(8, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("rr_target")
    plt.ylabel("ema_period")
    plt.title("Parameter Plateau Map")
    plt.colorbar(label="score")
    plt.tight_layout()
    plt.savefig("reports/plateau_map.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
