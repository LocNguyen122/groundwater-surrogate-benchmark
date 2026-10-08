"""Validate this private code snapshot, not excluded manuscript figures or cluster utilities."""
import json
from verify_snapshot import verify
from validate_v7_release import validate

if __name__ == "__main__":
    print(json.dumps({"snapshot": verify(), "scientific_scoring": validate()}, indent=2))
