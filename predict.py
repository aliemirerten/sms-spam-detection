"""Predict English SMS labels using the saved training-only pipeline."""

import argparse

import joblib

from data import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Classify an English SMS as ham or spam.")
    parser.add_argument("--message", help="Classify one message and exit.")
    args = parser.parse_args()
    model_path = OUTPUT_DIR / "spam_model.joblib"
    if not model_path.exists():
        parser.error("Model not found. Run train.py first.")
    model = joblib.load(model_path)

    def predict(message):
        if not message.strip():
            print("Please enter a non-empty English SMS.")
            return
        print(f"Prediction: {model.predict([message])[0]}")

    if args.message is not None:
        if not args.message.strip():
            parser.error("Message must not be empty.")
        predict(args.message)
        return
    print("Enter an English SMS. Type /quit to exit.")
    while True:
        try:
            message = input("SMS> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if message.strip().lower() == "/quit":
            print("Goodbye.")
            break
        predict(message)


if __name__ == "__main__":
    main()
