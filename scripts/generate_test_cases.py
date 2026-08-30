"""Generate N random valid CustomerProfile payloads as JSONL."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_predictor.utils.sample_generator import RandomCustomerGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    gen = RandomCustomerGenerator(seed=args.seed)
    for profile in gen.generate_batch(args.n):
        print(json.dumps(profile.model_dump(by_alias=True)))


if __name__ == "__main__":
    main()
