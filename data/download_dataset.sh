#!/usr/bin/env bash
set -euo pipefail

# —————————————————————————————————————————————
# Usage: download_dataset.sh <kaggle-dataset-slug>
#   e.g. download_dataset.sh sidharkal/sports-image-classification
#
# Works on:
#  • Linux (chmod +x; ./download_dataset.sh …)
#  • Windows Git Bash / WSL (bash download_dataset.sh …)
# —————————————————————————————————————————————

if [ $# -ne 1 ]; then
  cat <<-USAGE
Usage: $0 <kaggle-dataset-slug>
Example: $0 sidharkal/sports-image-classification
USAGE
  exit 1
fi

SLUG="$1"
# turn "user/dataset-name" → "user-dataset-name"
FOLDER="$(printf '%s' "$SLUG" | tr '/' '-')"

# download path is alongside this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/$FOLDER"

mkdir -p "$DEST_DIR"
echo "🔽 Downloading Kaggle dataset '$SLUG' → '$DEST_DIR' …"

# require kaggle CLI
if ! command -v kaggle >/dev/null 2>&1; then
  echo "❌ kaggle CLI not found. Install with:"
  echo "     pip install kaggle"
  echo "  and then place your ~/.kaggle/kaggle.json credentials file."
  exit 2
fi

# do the download + unzip
kaggle datasets download -d "$SLUG" \
   --unzip \
   -p "$DEST_DIR"

echo "✅ Done!  Files are in: $DEST_DIR"
