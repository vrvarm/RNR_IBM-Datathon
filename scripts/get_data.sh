#!/usr/bin/env bash
# Download the real Kaggle dataset into data/creditcard.csv.
# Needs Kaggle credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY in the environment.
#
# Note: `kaggle kernels pull gpreda/credit-card-fraud-detection-predictive-models` only downloads that
# author's notebook. The data it analyses is the dataset below.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
ls -lh data/creditcard.csv
