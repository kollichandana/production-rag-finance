"""Download a small set of 10-K filings from SEC EDGAR.

EDGAR requires a custom User-Agent identifying the requester. Replace the
default if you deploy this — see https://www.sec.gov/os/accessing-edgar-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from loguru import logger

USER_AGENT = "production-rag-demo your-email@example.com"

# Direct links to the 10-K HTML for FY2023 filings for a few large-cap tech names.
# These URLs come from each company's SEC EDGAR filing index.
DEFAULT_FILINGS = {
    "AAPL_Apple_2023_10K.html": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
    "MSFT_Microsoft_2023_10K.html": "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm",
    "GOOGL_Alphabet_2023_10K.html": "https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm",
    "AMZN_Amazon_2023_10K.html": "https://www.sec.gov/Archives/edgar/data/1018724/000101872424000008/amzn-20231231.htm",
    "META_Meta_2023_10K.html": "https://www.sec.gov/Archives/edgar/data/1326801/000132680124000012/meta-20231231.htm",
}


def download(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        for filename, url in DEFAULT_FILINGS.items():
            dest = out_dir / filename
            if dest.exists() and dest.stat().st_size > 50_000:
                logger.info(f"Skip (exists): {filename}")
                continue
            logger.info(f"Downloading {filename}")
            try:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                logger.info(f"  saved {len(resp.content):,} bytes")
            except Exception as e:
                logger.error(f"  failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()
    download(Path(args.out))


if __name__ == "__main__":
    main()
