"""
predict_house_price.py
──────────────────────
A simple command-line UI to predict California house prices
using the trained LinearRegression model.

Usage:
    python predict_house_price.py                  # interactive prompts
    python predict_house_price.py --demo           # run with a demo sample
"""

import pickle
import argparse
import sys
import pandas as pd

MODEL_PATH = "linear_regression_model.pkl"

FEATURE_INFO = {
    "MedInc":     ("Median household income",           "in $10,000 units (e.g., 5.0 = $50,000)"),
    "HouseAge":   ("Median house age",                  "in years (e.g., 25)"),
    "AveRooms":   ("Average rooms per household",       "e.g., 6.0"),
    "AveBedrms":  ("Average bedrooms per household",    "e.g., 1.0"),
    "Population": ("Block group population",            "e.g., 500"),
    "AveOccup":   ("Average household occupancy",       "e.g., 2.5"),
    "Latitude":   ("Block group latitude",              "degrees N (e.g., 37.0)"),
    "Longitude":  ("Block group longitude",             "degrees E, negative (e.g., -122.0)"),
}

DEMO_SAMPLE = {
    "MedInc": 8.0, "HouseAge": 25, "AveRooms": 6.0,
    "AveBedrms": 1.0, "Population": 500, "AveOccup": 2.5,
    "Latitude": 37.0, "Longitude": -122.0,
}


def load_model(path: str):
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["scaler"], bundle["features"]
    except FileNotFoundError:
        print(f"[ERROR] Model file '{path}' not found.")
        print("  → Run the Jupyter notebook first to generate the model pickle.")
        sys.exit(1)


def predict(model, scaler, features, inputs: dict) -> float:
    df = pd.DataFrame([inputs])[features]
    scaled = scaler.transform(df)
    return model.predict(scaled)[0]


def interactive_input(features: list) -> dict:
    print("\n" + "═" * 55)
    print("  California House Price Predictor")
    print("  Linear Regression Model · Maincrafts Technology")
    print("═" * 55)
    print("  Enter feature values below (press Enter for each)\n")

    inputs = {}
    for feat in features:
        label, hint = FEATURE_INFO.get(feat, (feat, ""))
        while True:
            try:
                val = input(f"  {label}\n    [{hint}]: ")
                inputs[feat] = float(val)
                break
            except ValueError:
                print("    ✗  Please enter a numeric value.\n")
    return inputs


def print_result(inputs: dict, pred: float):
    print("\n" + "─" * 55)
    print("  INPUT SUMMARY")
    print("─" * 55)
    for k, v in inputs.items():
        print(f"  {k:12s}: {v}")
    print("─" * 55)
    print(f"  PREDICTED MEDIAN HOUSE VALUE")
    print(f"  ➜  {pred:.4f} × $100,000")
    print(f"  ➜  ${pred * 100_000:,.0f}")
    print("─" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="California House Price Predictor")
    parser.add_argument("--demo", action="store_true", help="Run with a built-in demo sample")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the model pickle file")
    args = parser.parse_args()

    model, scaler, features = load_model(args.model)

    if args.demo:
        print("\n[DEMO MODE] Using sample input:")
        for k, v in DEMO_SAMPLE.items():
            print(f"  {k}: {v}")
        pred = predict(model, scaler, features, DEMO_SAMPLE)
        print_result(DEMO_SAMPLE, pred)
    else:
        inputs = interactive_input(features)
        pred = predict(model, scaler, features, inputs)
        print_result(inputs, pred)

    while True:
        again = input("  Predict another house? (y/n): ").strip().lower()
        if again == 'y':
            inputs = interactive_input(features)
            pred = predict(model, scaler, features, inputs)
            print_result(inputs, pred)
        else:
            print("  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
