"""
Automatic data download script for GDSC dataset.

This script downloads all required data files from the GDSC/CANCERRXGENE portal:
- Gene expression data (RMA normalized)
- Drug response data (IC50, AUC)
- Compound annotations
- Cell line details

The GDSC (Genomics of Drug Sensitivity in Cancer) project provides
comprehensive drug response data for cancer cell lines.

Author: Bachelor's Thesis Project
Date: 2026
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import logging

# Add project root to path to import config
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARCHIVE_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# GDSC DATA DOWNLOAD URLS
# =============================================================================

# Note: These URLs are from the GDSC/CANCERRXGENE bulk download page
# https://www.cancerrxgene.org/downloads/bulk_download

GDSC_DOWNLOADS = {
    # Gene expression data (RMA normalized)
    "gene_expression": {
        "url": "https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip",
        "filename": "Cell_line_RMA_proc_basalExp.txt.zip",
        "description": "Gene expression data (RMA normalized)",
        "size_mb": "~100 MB (compressed)"
    },

    # Drug response data (GDSC1) - Fitted dose response with IC50, AUC
    "drug_response_gdsc1": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx",
        "filename": "GDSC1_fitted_dose_response.xlsx",
        "description": "GDSC1 drug response data (IC50, AUC, fitted)",
        "size_mb": "~20 MB"
    },

    # Alternative: Raw drug response CSV (more detailed, larger file)
    "drug_response_raw": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_public_raw_data_27Oct23.csv.zip",
        "filename": "GDSC1_public_raw_data.csv.zip",
        "description": "GDSC1 public raw data (CSV format, compressed)",
        "size_mb": "~15 MB (compressed)"
    },

    # ANOVA results (optional - useful for feature selection)
    "anova_results": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/ANOVA_results_GDSC1_27Oct23.xlsx",
        "filename": "ANOVA_results_GDSC1.xlsx",
        "description": "ANOVA analysis of drug sensitivity associations",
        "size_mb": "~5 MB"
    },

    # Cell line details
    "cell_line_details": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx",
        "filename": "Cell_Lines_Details.xlsx",
        "description": "Cell line metadata (tissue, cancer type, etc.)",
        "size_mb": "~500 KB"
    },

    # Compound (drug) annotations
    "compound_annotations": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv",
        "filename": "screened_compounds.csv",
        "description": "Compound annotations (drug targets, pathways)",
        "size_mb": "~100 KB"
    },
}

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(url: str, output_path: Path, description: str = ""):
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        output_path: Path where to save the file
        description: Description of the file (for logging)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {description}")
        logger.info(f"URL: {url}")
        logger.info(f"Destination: {output_path}")

        # Make sure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Send GET request with stream=True to download in chunks
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise error for bad status codes

        # Get total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"✓ Successfully downloaded: {output_path.name}\n")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Failed to download {description}: {e}\n")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error downloading {description}: {e}\n")
        return False


def check_file_exists(filepath: Path, min_size_kb: int = 1) -> bool:
    """
    Check if a file exists and is not empty.

    Args:
        filepath: Path to check
        min_size_kb: Minimum file size in KB (default: 1KB)

    Returns:
        True if file exists and is large enough
    """
    if not filepath.exists():
        return False

    # Check file size (in bytes)
    file_size = filepath.stat().st_size
    min_size_bytes = min_size_kb * 1024

    return file_size >= min_size_bytes


def download_all_data(force_redownload: bool = False):
    """
    Download all GDSC data files.

    Args:
        force_redownload: If True, redownload even if files exist
    """
    logger.info("=" * 80)
    logger.info("GDSC DATA DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"Download directory: {ARCHIVE_DIR}\n")

    # Create archive directory if it doesn't exist
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Track download statistics
    total_files = len(GDSC_DOWNLOADS)
    downloaded = 0
    skipped = 0
    failed = 0

    # Download each file
    for dataset_name, info in GDSC_DOWNLOADS.items():
        output_path = ARCHIVE_DIR / info["filename"]

        # Check if file already exists
        if not force_redownload and check_file_exists(output_path):
            logger.info(f"⊙ File already exists: {info['filename']}")
            logger.info(f"  Skipping download. Use --force to redownload.\n")
            skipped += 1
            continue

        # Download the file
        success = download_file(
            url=info["url"],
            output_path=output_path,
            description=f"{info['description']} ({info['size_mb']})"
        )

        if success:
            downloaded += 1
        else:
            failed += 1

    # Print summary
    logger.info("=" * 80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files: {total_files}")
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Skipped (already exist): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 80)

    if failed == 0:
        logger.info("\n✓ All data files ready!")
        logger.info(f"Data location: {ARCHIVE_DIR}")
        logger.info("\nNext steps:")
        logger.info("1. Extract any .zip files if needed")
        logger.info("2. Run preprocessing: python scripts/01_preprocess_data.py")
    else:
        logger.warning(f"\n⚠ {failed} file(s) failed to download.")
        logger.warning("Please check the URLs or try again later.")
        logger.warning("You can also manually download files from:")
        logger.warning("https://www.cancerrxgene.org/downloads/bulk_download")


def extract_zip_files():
    """
    Extract any .zip files in the archive directory.
    """
    import zipfile

    logger.info("Checking for ZIP files to extract...")

    zip_files = list(ARCHIVE_DIR.glob("*.zip"))

    if not zip_files:
        logger.info("No ZIP files found.\n")
        return

    for zip_path in zip_files:
        try:
            logger.info(f"Extracting: {zip_path.name}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to archive directory
                zip_ref.extractall(ARCHIVE_DIR)

            logger.info(f"✓ Extracted: {zip_path.name}\n")

        except Exception as e:
            logger.error(f"✗ Failed to extract {zip_path.name}: {e}\n")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function to run data download.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GDSC dataset files automatically"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force redownload even if files exist'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract ZIP files after download'
    )

    args = parser.parse_args()

    # Download all data
    download_all_data(force_redownload=args.force)

    # Extract ZIP files if requested
    if args.extract:
        extract_zip_files()


if __name__ == "__main__":
    main()
