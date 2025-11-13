from __future__ import annotations

from bailiff.metrics.tone import FrozenToneClassifier, calibrate_frozen_tone_model


def main():
    clf = FrozenToneClassifier()
    while True:
        statement = input("Input sentence to score ('exit' to exit): ")
        if(statement == "exit"):
            break
        print(f"Score: {clf.score(statement)}")


if __name__ == "__main__":
    main()