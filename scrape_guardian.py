import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.theguardian.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}
_thread_local = threading.local()


def normalize_guardian_url(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        url = raw
    else:
        url = urljoin(BASE_URL, raw)
    return url.split("#")[0]


def extract_article(url: str, session: requests.Session, timeout: int = 20) -> dict:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    selectors = [
        "div[data-gu-name='body'] p",
        "main article p",
        "article p",
        "main p",
    ]

    paragraphs = []
    for selector in selectors:
        nodes = soup.select(selector)
        if len(nodes) >= 3:
            paragraphs = [n.get_text(" ", strip=True) for n in nodes]
            break

    if not paragraphs:
        paragraphs = [n.get_text(" ", strip=True) for n in soup.find_all("p")]

    text = "\n".join([p for p in paragraphs if p and len(p.split()) >= 3]).strip()

    return {
        "title": title,
        "text": text,
        "word_count": len(text.split()) if text else 0,
    }


def get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        sess = requests.Session()
        sess.headers.update(HEADERS)
        _thread_local.session = sess
    return _thread_local.session


def fetch_with_retries(
    url: str,
    label: str,
    idx: int,
    total: int,
    timeout: int,
    retries: int,
    sleep_sec: float,
    min_words: int,
) -> dict | None:
    for attempt in range(retries + 1):
        try:
            session = get_thread_session()
            article = extract_article(url, session, timeout=timeout)
            if article["word_count"] < min_words:
                print(f"[{label}] {idx}/{total} SKIP(short) -> {url}")
                return None
            print(f"[{label}] {idx}/{total} OK -> {url}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return {
                "label": label,
                "url": url,
                "title": article["title"],
                "text": article["text"],
                "word_count": article["word_count"],
            }
        except Exception as exc:
            if attempt < retries:
                backoff = 0.2 * (attempt + 1)
                time.sleep(backoff)
            else:
                print(f"[{label}] {idx}/{total} FAIL -> {url} ({exc})")
    return None


def load_urls(csv_path: Path, max_rows: int | None = None) -> list[str]:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError(f"Missing 'url' column in {csv_path}")

    urls = df["url"].dropna().astype(str).map(normalize_guardian_url)
    urls = urls[urls != ""].drop_duplicates().tolist()
    if max_rows:
        urls = urls[:max_rows]
    return urls


def scrape_from_csv(
    csv_path: Path,
    label: str,
    max_rows: int | None = None,
    sleep_sec: float = 0.05,
    workers: int = 12,
    timeout: int = 20,
    retries: int = 2,
    min_words: int = 80,
) -> pd.DataFrame:
    urls = load_urls(csv_path, max_rows=max_rows)
    rows = []
    total = len(urls)
    if total == 0:
        return pd.DataFrame(rows)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fetch_with_retries, url, label, i, total, timeout, retries, sleep_sec, min_words
            )
            for i, url in enumerate(urls, start=1)
        ]
        for future in as_completed(futures):
            item = future.result()
            if item is not None:
                rows.append(item)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Guardian politics/sport articles from URL CSV files.")
    parser.add_argument("--politics_csv", default="guardian_politics_links.csv")
    parser.add_argument("--sports_csv", default="guardian_sports_links.csv")
    parser.add_argument("--output", default="data/raw/guardian_articles.csv")
    parser.add_argument("--max_rows_per_class", type=int, default=1500)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min_words", type=int, default=80)
    parser.add_argument("--dry_run_count", type=int, default=0)
    parser.add_argument("--dry_run_output", default="data/raw/dry_run_preview.csv")
    args = parser.parse_args()

    if args.dry_run_count > 0:
        dry_politics_df = scrape_from_csv(
            Path(args.politics_csv),
            "politics",
            max_rows=args.dry_run_count,
            sleep_sec=args.sleep,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            min_words=args.min_words,
        )
        dry_sports_df = scrape_from_csv(
            Path(args.sports_csv),
            "sport",
            max_rows=args.dry_run_count,
            sleep_sec=args.sleep,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            min_words=args.min_words,
        )
        dry_df = pd.concat([dry_politics_df, dry_sports_df], ignore_index=True)
        dry_df = dry_df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["text"])
        dry_df = dry_df[dry_df["text"].str.len() > 0].reset_index(drop=True)
        dry_df["text_preview"] = dry_df["text"].str.slice(0, 300)

        dry_out_path = Path(args.dry_run_output)
        dry_out_path.parent.mkdir(parents=True, exist_ok=True)
        dry_df[["label", "url", "title", "word_count", "text_preview"]].to_csv(dry_out_path, index=False)

        print("\nDry run saved:", dry_out_path)
        print("Dry run class counts:")
        print(dry_df["label"].value_counts())
        print("Dry run total:", len(dry_df))
        return

    max_rows = args.max_rows_per_class if args.max_rows_per_class > 0 else 1500

    politics_df = scrape_from_csv(
        Path(args.politics_csv),
        "politics",
        max_rows=max_rows,
        sleep_sec=args.sleep,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        min_words=args.min_words,
    )
    sports_df = scrape_from_csv(
        Path(args.sports_csv),
        "sport",
        max_rows=max_rows,
        sleep_sec=args.sleep,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        min_words=args.min_words,
    )

    all_df = pd.concat([politics_df, sports_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["text"])
    all_df = all_df[all_df["text"].str.len() > 0].reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("Class counts:")
    print(all_df["label"].value_counts())
    print("Total:", len(all_df))


if __name__ == "__main__":
    main()
