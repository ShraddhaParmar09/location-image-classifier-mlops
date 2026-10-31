import io
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from remotezip import RemoteZip
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
SELECTION = BASE / "data" / "metadata" / "pilot_selection.csv"
IMAGE_ROOT = BASE / "data" / "images"
REPORT = BASE / "reports" / "image_download_report.csv"

REPO_URL = (
    "https://huggingface.co/datasets/osv5m/osv5m/"
    "resolve/main/images/train/{shard:02d}.zip"
)

if not SELECTION.exists():
    raise FileNotFoundError(SELECTION)

if REPORT.exists():
    raise FileExistsError(
        f"Report already exists: {REPORT}. "
        "Move it aside before starting a new report."
    )

df = pd.read_csv(SELECTION, dtype={"id": str})
df["country"] = df["country"].astype(str)

if len(df) != 1000:
    raise ValueError(f"Expected 1000 selected images, found {len(df)}")

# Map image ID to its country and destination.
wanted = {}

for row in df.itertuples(index=False):
    image_id = str(row.id)
    country = str(row.country)
    destination = IMAGE_ROOT / country / f"{image_id}.jpg"

    if image_id in wanted:
        raise ValueError(f"Duplicate image ID: {image_id}")

    wanted[image_id] = {
        "country": country,
        "destination": destination,
    }

# Validate existing images; never overwrite them.
found = set()
for image_id, info in wanted.items():
    destination = info["destination"]

    if destination.exists():
        try:
            with Image.open(destination) as image:
                image.verify()
            found.add(image_id)
        except Exception:
            print(f"Existing invalid image preserved: {destination}")

remaining = set(wanted) - found

print(f"Selected images: {len(wanted)}")
print(f"Already recovered and valid: {len(found)}")
print(f"Images remaining: {len(remaining)}")
print("Scanning 98 remote ZIP shards...")

records = []
errors = []
downloaded = 0

for shard in range(98):
    if not remaining:
        break

    url = REPO_URL.format(shard=shard)
    print(f"\nShard {shard:02d}/97 | Remaining: {len(remaining)}")

    for attempt in range(3):
        try:
            with RemoteZip(url) as archive:
                for member in archive.namelist():
                    if not member.lower().endswith(".jpg"):
                        continue

                    image_id = member.rsplit("/", 1)[-1][:-4]

                    if image_id not in remaining:
                        continue

                    destination = wanted[image_id]["destination"]
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    # Protect any file that appeared after the initial scan.
                    if destination.exists():
                        try:
                            with Image.open(destination) as image:
                                image.verify()
                            found.add(image_id)
                            remaining.remove(image_id)
                            continue
                        except Exception:
                            print(
                                "Invalid existing file preserved:",
                                destination
                            )
                            continue

                    try:
                        image_bytes = archive.read(member)

                        with Image.open(io.BytesIO(image_bytes)) as image:
                            image.verify()

                        # Write to a temporary file first.
                        temp_path = destination.with_name(
                            destination.name + ".download_tmp"
                        )

                        if temp_path.exists():
                            raise FileExistsError(temp_path)

                        temp_path.write_bytes(image_bytes)

                        # Validate the saved file too.
                        with Image.open(temp_path) as image:
                            image.verify()

                        # Do not replace an existing destination.
                        if destination.exists():
                            raise FileExistsError(destination)

                        temp_path.rename(destination)

                        found.add(image_id)
                        remaining.remove(image_id)
                        downloaded += 1

                        records.append({
                            "id": image_id,
                            "country": wanted[image_id]["country"],
                            "shard": shard,
                            "status": "downloaded",
                            "path": str(destination),
                        })

                        print(
                            f"Recovered {downloaded} new images; "
                            f"{len(remaining)} remaining"
                        )

                    except Exception as exc:
                        errors.append({
                            "id": image_id,
                            "country": wanted[image_id]["country"],
                            "shard": shard,
                            "error": str(exc),
                        })

            break

        except Exception as exc:
            print(f"Shard {shard:02d} attempt {attempt + 1} failed: {exc}")

            if attempt == 2:
                errors.append({
                    "id": "",
                    "country": "",
                    "shard": shard,
                    "error": str(exc),
                })
            else:
                time.sleep(3 * (attempt + 1))

# Build a complete status report.
for image_id, info in wanted.items():
    if image_id in found:
        continue

    records.append({
        "id": image_id,
        "country": info["country"],
        "shard": "",
        "status": "missing",
        "path": "",
    })

pd.DataFrame(records).to_csv(REPORT, index=False)

print("\n========== RECOVERY SUMMARY ==========")
print("Selected:", len(wanted))
print("Valid recovered:", len(found))
print("Newly downloaded:", downloaded)
print("Still missing:", len(remaining))
print("Shard/network errors:", len(errors))
print("Report:", REPORT)

if errors:
    error_path = BASE / "reports" / "image_download_errors.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print("Error details:", error_path)
