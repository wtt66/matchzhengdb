from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@dataclass
class TextRecord:
    platform: str
    record_type: str
    keyword: str = ""
    record_id: str = ""
    parent_id: str = ""
    product_id: str = ""
    product_name: str = ""
    brand: str = ""
    title: str = ""
    content: str = ""
    rating: float | None = None
    publish_time: str = ""
    likes: int | None = None
    replies: int | None = None
    tags: str = ""
    url: str = ""
    user_name: str = ""
    user_id: str = ""
    price_text: str = ""
    source_query: str = ""
    raw_json: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if data["rating"] is None:
            data["rating"] = ""
        if data["likes"] is None:
            data["likes"] = ""
        if data["replies"] is None:
            data["replies"] = ""
        return data


def build_session(cookie_string: str = "", extra_headers: dict[str, str] | None = None) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(DEFAULT_HEADERS)
    if extra_headers:
        session.headers.update(extra_headers)
    if cookie_string:
        session.headers["Cookie"] = cookie_string
    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    timeout: int = 20,
    max_retries: int = 3,
) -> requests.Response:
    last_error: Exception | None = None
    candidate_urls = [url]
    if url.startswith("https://"):
        candidate_urls.append("http://" + url[len("https://") :])

    for candidate_url in candidate_urls:
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(candidate_url, params=params, timeout=timeout, verify=False)
                response.raise_for_status()
                return response
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(1.5 * attempt)
    if last_error:
        raise last_error
    raise RuntimeError("request_with_retry failed without an explicit exception")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def parse_count(raw: object) -> int | None:
    if raw is None:
        return None
    text = normalize_text(str(raw)).lower()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    number = float(match.group(1))
    if "万" in text or "w" in text:
        number *= 10000
    return int(number)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_records(records: Iterable[TextRecord], output_path: Path) -> pd.DataFrame:
    rows = [record.to_dict() for record in records]
    frame = pd.DataFrame(rows)
    ensure_parent(output_path)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as fh:
            for row in frame.to_dict(orient="records"):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fragrance market texts from multiple sources and normalize them for BERTopic."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    jd_parser = subparsers.add_parser("jd", help="Collect JD product reviews from the public comment endpoint.")
    jd_parser.add_argument("--sku", nargs="+", required=True, help="One or more JD SKU ids.")
    jd_parser.add_argument("--pages", type=int, default=10, help="Pages per SKU.")
    jd_parser.add_argument("--page-size", type=int, default=10, help="Records per page.")
    jd_parser.add_argument("--sort-type", type=int, default=5, help="5 means newest first on most products.")
    jd_parser.add_argument("--delay", type=float, default=1.8, help="Base wait seconds between requests.")
    jd_parser.add_argument("--keyword", default="", help="Optional research keyword for later grouping.")
    jd_parser.add_argument("--output", required=True, help="CSV or JSONL output path.")

    jd_keyword_parser = subparsers.add_parser(
        "jd-keyword",
        help="Search JD products by keyword and then collect reviews for the discovered SKUs.",
    )
    jd_keyword_parser.add_argument("--keyword", nargs="+", required=True, help="One or more JD search keywords.")
    jd_keyword_parser.add_argument("--search-pages", type=int, default=2, help="JD search result pages per keyword.")
    jd_keyword_parser.add_argument("--max-skus", type=int, default=5, help="Maximum SKUs collected per keyword.")
    jd_keyword_parser.add_argument("--review-pages", type=int, default=8, help="Review pages per SKU.")
    jd_keyword_parser.add_argument("--page-size", type=int, default=10, help="Review records per page.")
    jd_keyword_parser.add_argument("--sort-type", type=int, default=5, help="5 means newest first on most products.")
    jd_keyword_parser.add_argument("--delay", type=float, default=1.8, help="Base wait seconds between requests.")
    jd_keyword_parser.add_argument("--output", required=True, help="CSV or JSONL output path.")

    xhs_parser = subparsers.add_parser(
        "xhs",
        help="Collect Xiaohongshu notes and comments with Playwright. Manual login is expected on first run.",
    )
    xhs_parser.add_argument("--keyword", nargs="+", required=True, help="One or more search keywords.")
    xhs_parser.add_argument("--max-notes", type=int, default=30, help="Maximum notes per keyword.")
    xhs_parser.add_argument("--max-comments", type=int, default=20, help="Maximum comments per note.")
    xhs_parser.add_argument("--scroll-rounds", type=int, default=12, help="Maximum search-result scroll rounds.")
    xhs_parser.add_argument("--delay", type=float, default=2.5, help="Base wait seconds between page actions.")
    xhs_parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    xhs_parser.add_argument(
        "--storage-state",
        default="data/raw_scraped/xhs_storage_state.json",
        help="Playwright storage-state path for cookie/session reuse.",
    )
    xhs_parser.add_argument("--output", required=True, help="CSV or JSONL output path.")

    generic_parser = subparsers.add_parser(
        "generic",
        help="Collect public-page article/review content using a JSON selector config.",
    )
    generic_parser.add_argument("--config", required=True, help="JSON config path.")
    generic_parser.add_argument("--keyword", nargs="+", required=True, help="One or more search keywords.")
    generic_parser.add_argument("--max-links", type=int, default=20, help="Maximum detail links per keyword.")
    generic_parser.add_argument("--max-comments", type=int, default=10, help="Maximum comments per detail page.")
    generic_parser.add_argument("--delay", type=float, default=1.5, help="Base wait seconds between requests.")
    generic_parser.add_argument("--cookie", default="", help="Optional cookie string for pages that require login.")
    generic_parser.add_argument("--output", required=True, help="CSV or JSONL output path.")

    url_list_parser = subparsers.add_parser(
        "url-list",
        help="Collect public pages from a prepared URL list using a JSON selector config.",
    )
    url_list_parser.add_argument("--config", required=True, help="JSON config path.")
    url_list_parser.add_argument("--urls-file", required=True, help="Text file with one URL per line.")
    url_list_parser.add_argument("--keyword", default="", help="Optional grouping keyword for later analysis.")
    url_list_parser.add_argument("--max-comments", type=int, default=0, help="Maximum comments per detail page.")
    url_list_parser.add_argument("--delay", type=float, default=1.2, help="Base wait seconds between requests.")
    url_list_parser.add_argument("--cookie", default="", help="Optional cookie string for pages that require login.")
    url_list_parser.add_argument("--output", required=True, help="CSV or JSONL output path.")

    return parser.parse_args()


def jd_reviews(
    sku_ids: list[str],
    *,
    keyword: str,
    pages: int,
    page_size: int,
    sort_type: int,
    delay: float,
    sku_meta: dict[str, dict[str, str]] | None = None,
) -> list[TextRecord]:
    session = build_session()
    endpoint = "https://club.jd.com/comment/productPageComments.action"
    records: list[TextRecord] = []
    sku_meta = sku_meta or {}

    for sku in sku_ids:
        for page in range(pages):
            params = {
                "productId": sku,
                "score": 0,
                "sortType": sort_type,
                "page": page,
                "pageSize": page_size,
                "isShadowSku": 0,
                "fold": 1,
            }
            response = request_with_retry(session, endpoint, params=params)
            payload = response.json()
            comments = payload.get("comments", []) or []
            if not comments:
                break

            for item in comments:
                content = normalize_text(item.get("content", ""))
                if not content:
                    continue
                record = TextRecord(
                    platform="jd",
                    record_type="review",
                    keyword=keyword,
                    source_query=keyword or sku,
                    record_id=str(item.get("id") or item.get("guid") or ""),
                    product_id=sku,
                    product_name=normalize_text(item.get("referenceName", "")) or sku_meta.get(sku, {}).get("product_name", ""),
                    title=normalize_text(item.get("referenceName", "")) or sku_meta.get(sku, {}).get("product_name", ""),
                    content=content,
                    rating=float(item.get("score")) if item.get("score") is not None else None,
                    publish_time=str(item.get("creationTime", "")),
                    likes=parse_count(item.get("usefulVoteCount")),
                    replies=parse_count(item.get("replyCount")),
                    url=sku_meta.get(sku, {}).get("url", f"https://item.jd.com/{sku}.html"),
                    user_name=normalize_text(item.get("nickname", "")),
                    user_id=str(item.get("id", "")),
                    price_text=normalize_text(item.get("productColor", "")) or sku_meta.get(sku, {}).get("price_text", ""),
                    raw_json=json.dumps(item, ensure_ascii=False),
                )
                records.append(record)

                follow_up = normalize_text(item.get("afterUserComment", {}).get("content", ""))
                if follow_up:
                    records.append(
                        TextRecord(
                            platform="jd",
                            record_type="follow_up",
                            keyword=keyword,
                            source_query=keyword or sku,
                            record_id=f"{record.record_id}_followup",
                            parent_id=record.record_id,
                            product_id=sku,
                            product_name=record.product_name,
                            title=record.title,
                            content=follow_up,
                            rating=record.rating,
                            publish_time=str(item.get("afterUserComment", {}).get("created", "")),
                            likes=record.likes,
                            replies=record.replies,
                            url=record.url,
                            user_name=record.user_name,
                            user_id=record.user_id,
                            price_text=record.price_text,
                            raw_json=json.dumps(item.get("afterUserComment", {}), ensure_ascii=False),
                        )
                    )
            time.sleep(delay + random.uniform(0.2, 0.8))
    return records


def jd_search_products(
    keywords: list[str],
    *,
    search_pages: int,
    max_skus: int,
    delay: float,
) -> list[dict[str, str]]:
    session = build_session()
    search_url = "https://search.jd.com/Search"
    products: list[dict[str, str]] = []
    seen: set[str] = set()

    for keyword in keywords:
        keyword_count = 0
        for page_idx in range(search_pages):
            params = {"keyword": keyword, "enc": "utf-8", "page": page_idx * 2 + 1}
            response = request_with_retry(session, search_url, params=params)
            soup = BeautifulSoup(response.text, "html.parser")
            nodes = soup.select("li.gl-item[data-sku]")
            for node in nodes:
                sku = normalize_text(node.get("data-sku", ""))
                if not sku or sku in seen:
                    continue
                title_node = node.select_one(".p-name em, .p-name-type-2 em, .p-name a em")
                title = normalize_text(title_node.get_text(" ", strip=True)) if title_node else ""
                if not title:
                    continue
                price_node = node.select_one(".p-price i, .p-price strong i")
                price_text = normalize_text(price_node.get_text(" ", strip=True)) if price_node else ""
                products.append(
                    {
                        "keyword": keyword,
                        "sku": sku,
                        "product_name": title,
                        "price_text": price_text,
                        "url": f"https://item.jd.com/{sku}.html",
                    }
                )
                seen.add(sku)
                keyword_count += 1
                if keyword_count >= max_skus:
                    break
            if keyword_count >= max_skus:
                break
            time.sleep(delay + random.uniform(0.1, 0.5))
    return products


def jd_reviews_by_keyword(
    keywords: list[str],
    *,
    search_pages: int,
    max_skus: int,
    review_pages: int,
    page_size: int,
    sort_type: int,
    delay: float,
) -> list[TextRecord]:
    products = jd_search_products(
        keywords,
        search_pages=search_pages,
        max_skus=max_skus,
        delay=delay,
    )
    if not products:
        return []

    grouped: dict[str, list[dict[str, str]]] = {}
    for product in products:
        grouped.setdefault(product["keyword"], []).append(product)

    records: list[TextRecord] = []
    for keyword, items in grouped.items():
        sku_meta = {item["sku"]: item for item in items}
        records.extend(
            jd_reviews(
                [item["sku"] for item in items],
                keyword=keyword,
                pages=review_pages,
                page_size=page_size,
                sort_type=sort_type,
                delay=delay,
                sku_meta=sku_meta,
            )
        )
    return records


def extract_first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            if node.name == "meta" and node.get("content"):
                return normalize_text(node.get("content", ""))
            return normalize_text(node.get_text(" ", strip=True))
    return ""


def extract_many_texts(soup: BeautifulSoup, selectors: list[str], limit: int | None = None) -> list[str]:
    values: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = normalize_text(node.get_text(" ", strip=True))
            if text and text not in values:
                values.append(text)
                if limit and len(values) >= limit:
                    return values
    return values


def extract_content_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    blocks: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        for node in soup.select(selector):
            text = normalize_text(node.get_text(" ", strip=True))
            if not text or text in seen:
                continue
            seen.add(text)
            blocks.append(text)
    return normalize_text(" ".join(blocks))


def load_html_with_playwright(url: str, storage_state: Path, headless: bool, delay: float) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is required for the xhs command. Install it with: pip install playwright && playwright install chromium"
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context_kwargs: dict[str, object] = {}
        if storage_state.exists():
            context_kwargs["storage_state"] = str(storage_state)
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(int(delay * 1000))
        html = page.content()
        context.storage_state(path=str(storage_state))
        browser.close()
    return html


def bootstrap_xhs_login(storage_state: Path, headless: bool) -> None:
    if storage_state.exists():
        return
    if headless:
        raise RuntimeError("Xiaohongshu first-run login requires a visible browser. Remove --headless for the first run.")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is required for the xhs command. Install it with: pip install playwright && playwright install chromium"
        ) from exc

    storage_state.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.xiaohongshu.com", wait_until="domcontentloaded", timeout=90000)
        print("Please finish Xiaohongshu login in the opened browser, then press Enter here to continue.")
        input()
        context.storage_state(path=str(storage_state))
        browser.close()


def discover_xhs_note_urls(
    keywords: list[str],
    *,
    storage_state: Path,
    max_notes: int,
    scroll_rounds: int,
    delay: float,
    headless: bool,
) -> dict[str, list[str]]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright is required for the xhs command. Install it with: pip install playwright && playwright install chromium"
        ) from exc

    discovered: dict[str, list[str]] = {}
    bootstrap_xhs_login(storage_state, headless)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=str(storage_state))
        page = context.new_page()

        for keyword in keywords:
            search_url = (
                "https://www.xiaohongshu.com/search_result"
                f"?keyword={quote_plus(keyword)}&source=web_explore_feed"
            )
            page.goto(search_url, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(int(delay * 1000))
            links: list[str] = []
            idle_rounds = 0

            for _ in range(scroll_rounds):
                found = page.eval_on_selector_all(
                    "a[href]",
                    """
                    nodes => nodes
                        .map(node => node.href)
                        .filter(href => /xiaohongshu\\.com\\/(explore|discovery\\/item)\\//.test(href))
                    """,
                )
                current_count = len(links)
                for link in found:
                    if link not in links:
                        links.append(link)
                if len(links) >= max_notes:
                    break
                page.mouse.wheel(0, 2400)
                page.wait_for_timeout(int((delay + random.uniform(0.4, 1.0)) * 1000))
                if len(links) == current_count:
                    idle_rounds += 1
                else:
                    idle_rounds = 0
                if idle_rounds >= 3:
                    break
            discovered[keyword] = links[:max_notes]
        context.storage_state(path=str(storage_state))
        browser.close()
    return discovered


def xhs_records(
    keywords: list[str],
    *,
    storage_state: Path,
    max_notes: int,
    max_comments: int,
    scroll_rounds: int,
    delay: float,
    headless: bool,
) -> list[TextRecord]:
    note_map = discover_xhs_note_urls(
        keywords,
        storage_state=storage_state,
        max_notes=max_notes,
        scroll_rounds=scroll_rounds,
        delay=delay,
        headless=headless,
    )
    records: list[TextRecord] = []

    for keyword, urls in note_map.items():
        for url in urls:
            html = load_html_with_playwright(url, storage_state, headless, delay)
            soup = BeautifulSoup(html, "html.parser")
            title = ""
            if soup.title:
                title = normalize_text(soup.title.get_text())
            title = extract_first_text(soup, ["#detail-title", "h1", "title"]) or title
            content = extract_first_text(
                soup,
                ["#detail-desc", ".note-content", ".desc", "meta[name='description']"],
            )
            note_id_match = re.search(r"/(?:explore|item)/([a-zA-Z0-9]+)", url)
            note_id = note_id_match.group(1) if note_id_match else ""
            tags = extract_many_texts(
                soup,
                ["a[href*='search_result']", ".tag", ".topic-tag"],
                limit=8,
            )
            author = extract_first_text(
                soup,
                [".author-name", ".user-name", "a[href*='/user/profile/']"],
            )
            publish_time = extract_first_text(
                soup,
                [".date", ".publish-time", "time"],
            )
            likes = parse_count(extract_first_text(soup, [".like-wrapper .count", ".interactions .count"]))

            if content or title:
                records.append(
                    TextRecord(
                        platform="xiaohongshu",
                        record_type="note",
                        keyword=keyword,
                        source_query=keyword,
                        record_id=note_id,
                        title=title,
                        content=content,
                        publish_time=publish_time,
                        likes=likes,
                        tags="|".join(tags),
                        url=url,
                        user_name=author,
                    )
                )

            comment_nodes = soup.select("div.comment-item, div.parent-comment, div[class*='comment-item']")
            seen_text: set[str] = set()
            for idx, node in enumerate(comment_nodes[:max_comments], start=1):
                comment_text = extract_first_text(node, [".content", ".comment-content", ".note-text", "span"])
                if not comment_text or comment_text in seen_text:
                    continue
                seen_text.add(comment_text)
                comment_author = extract_first_text(node, [".author", ".name", "a"])
                comment_time = extract_first_text(node, [".time", ".date"])
                comment_like = parse_count(extract_first_text(node, [".like", ".count"]))
                records.append(
                    TextRecord(
                        platform="xiaohongshu",
                        record_type="comment",
                        keyword=keyword,
                        source_query=keyword,
                        record_id=f"{note_id}_comment_{idx}",
                        parent_id=note_id,
                        title=title,
                        content=comment_text,
                        publish_time=comment_time,
                        likes=comment_like,
                        tags="|".join(tags),
                        url=url,
                        user_name=comment_author,
                    )
                )
            time.sleep(delay + random.uniform(0.3, 1.0))
    return records


def select_one(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        if selector.startswith("meta:"):
            meta_key = selector.split(":", 1)[1]
            node = soup.find("meta", attrs={"name": meta_key}) or soup.find("meta", attrs={"property": meta_key})
            if node and node.get("content"):
                return normalize_text(node["content"])
        else:
            node = soup.select_one(selector)
            if node:
                return normalize_text(node.get_text(" ", strip=True))
    return ""


def discover_generic_links(
    session: requests.Session,
    config: dict[str, object],
    keyword: str,
    max_links: int,
) -> list[str]:
    search_url = str(config["search_url"]).format(keyword=quote_plus(keyword))
    response = request_with_retry(session, search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    selector = str(config["list_link_selector"])
    links: list[str] = []
    for node in soup.select(selector):
        href = (node.get("href") or "").strip()
        if not href:
            continue
        full_url = urljoin(search_url, href)
        if full_url not in links:
            links.append(full_url)
        if len(links) >= max_links:
            break
    return links


def generic_records(
    *,
    config_path: Path,
    keywords: list[str],
    max_links: int,
    max_comments: int,
    delay: float,
    cookie_string: str,
) -> list[TextRecord]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    session = build_session(cookie_string=cookie_string, extra_headers=config.get("headers"))
    detail_cfg = config.get("detail", {})
    records: list[TextRecord] = []

    for keyword in keywords:
        detail_urls = discover_generic_links(session, config, keyword, max_links)
        for detail_url in detail_urls:
            response = request_with_retry(session, detail_url)
            soup = BeautifulSoup(response.text, "html.parser")

            title = select_one(soup, detail_cfg.get("title_selectors", ["title"]))
            content = extract_content_text(soup, detail_cfg.get("content_selectors", []))
            publish_time = select_one(soup, detail_cfg.get("publish_time_selectors", []))
            author = select_one(soup, detail_cfg.get("author_selectors", []))
            tags = extract_many_texts(soup, detail_cfg.get("tag_selectors", []), limit=10)
            likes = parse_count(select_one(soup, detail_cfg.get("like_selectors", [])))

            base_id = re.sub(r"\W+", "_", detail_url).strip("_")
            if content or title:
                records.append(
                    TextRecord(
                        platform=str(config.get("platform", "generic")),
                        record_type="post",
                        keyword=keyword,
                        source_query=keyword,
                        record_id=base_id,
                        title=title,
                        content=content,
                        publish_time=publish_time,
                        likes=likes,
                        tags="|".join(tags),
                        url=detail_url,
                        user_name=author,
                    )
                )

            comment_selector = detail_cfg.get("comment_selector")
            if comment_selector:
                for idx, node in enumerate(soup.select(comment_selector)[:max_comments], start=1):
                    comment_text = normalize_text(node.get_text(" ", strip=True))
                    if not comment_text:
                        continue
                    comment_author = select_one(node, detail_cfg.get("comment_author_selectors", []))
                    comment_time = select_one(node, detail_cfg.get("comment_time_selectors", []))
                    comment_like = parse_count(select_one(node, detail_cfg.get("comment_like_selectors", [])))
                    records.append(
                        TextRecord(
                            platform=str(config.get("platform", "generic")),
                            record_type="comment",
                            keyword=keyword,
                            source_query=keyword,
                            record_id=f"{base_id}_comment_{idx}",
                            parent_id=base_id,
                            title=title,
                            content=comment_text,
                            publish_time=comment_time,
                            likes=comment_like,
                            tags="|".join(tags),
                            url=detail_url,
                            user_name=comment_author,
                        )
                    )
            time.sleep(delay + random.uniform(0.1, 0.6))
    return records


def records_from_url_list(
    *,
    config_path: Path,
    urls_file: Path,
    keyword: str,
    max_comments: int,
    delay: float,
    cookie_string: str,
) -> list[TextRecord]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    detail_cfg = config.get("detail", {})
    session = build_session(cookie_string=cookie_string, extra_headers=config.get("headers"))
    urls = [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    records: list[TextRecord] = []
    for detail_url in urls:
        response = request_with_retry(session, detail_url)
        soup = BeautifulSoup(response.text, "html.parser")

        title = select_one(soup, detail_cfg.get("title_selectors", ["title"]))
        content = extract_content_text(soup, detail_cfg.get("content_selectors", []))
        publish_time = select_one(soup, detail_cfg.get("publish_time_selectors", []))
        author = select_one(soup, detail_cfg.get("author_selectors", []))
        tags = extract_many_texts(soup, detail_cfg.get("tag_selectors", []), limit=10)
        likes = parse_count(select_one(soup, detail_cfg.get("like_selectors", [])))
        base_id = re.sub(r"\W+", "_", detail_url).strip("_")

        if content or title:
            records.append(
                TextRecord(
                    platform=str(config.get("platform", "url_list")),
                    record_type="post",
                    keyword=keyword,
                    source_query=keyword or str(config.get("platform", "url_list")),
                    record_id=base_id,
                    title=title,
                    content=content,
                    publish_time=publish_time,
                    likes=likes,
                    tags="|".join(tags),
                    url=detail_url,
                    user_name=author,
                )
            )

        comment_selector = detail_cfg.get("comment_selector")
        if comment_selector and max_comments > 0:
            for idx, node in enumerate(soup.select(comment_selector)[:max_comments], start=1):
                comment_text = normalize_text(node.get_text(" ", strip=True))
                if not comment_text:
                    continue
                comment_author = select_one(node, detail_cfg.get("comment_author_selectors", []))
                comment_time = select_one(node, detail_cfg.get("comment_time_selectors", []))
                comment_like = parse_count(select_one(node, detail_cfg.get("comment_like_selectors", [])))
                records.append(
                    TextRecord(
                        platform=str(config.get("platform", "url_list")),
                        record_type="comment",
                        keyword=keyword,
                        source_query=keyword or str(config.get("platform", "url_list")),
                        record_id=f"{base_id}_comment_{idx}",
                        parent_id=base_id,
                        title=title,
                        content=comment_text,
                        publish_time=comment_time,
                        likes=comment_like,
                        tags="|".join(tags),
                        url=detail_url,
                        user_name=comment_author,
                    )
                )
        time.sleep(delay + random.uniform(0.1, 0.5))
    return records


def main() -> None:
    args = parse_args()
    output = Path(args.output)

    if args.command == "jd":
        records = jd_reviews(
            args.sku,
            keyword=args.keyword,
            pages=args.pages,
            page_size=args.page_size,
            sort_type=args.sort_type,
            delay=args.delay,
        )
    elif args.command == "jd-keyword":
        records = jd_reviews_by_keyword(
            args.keyword,
            search_pages=args.search_pages,
            max_skus=args.max_skus,
            review_pages=args.review_pages,
            page_size=args.page_size,
            sort_type=args.sort_type,
            delay=args.delay,
        )
    elif args.command == "xhs":
        records = xhs_records(
            args.keyword,
            storage_state=Path(args.storage_state),
            max_notes=args.max_notes,
            max_comments=args.max_comments,
            scroll_rounds=args.scroll_rounds,
            delay=args.delay,
            headless=args.headless,
        )
    elif args.command == "url-list":
        records = records_from_url_list(
            config_path=Path(args.config),
            urls_file=Path(args.urls_file),
            keyword=args.keyword,
            max_comments=args.max_comments,
            delay=args.delay,
            cookie_string=args.cookie,
        )
    else:
        records = generic_records(
            config_path=Path(args.config),
            keywords=args.keyword,
            max_links=args.max_links,
            max_comments=args.max_comments,
            delay=args.delay,
            cookie_string=args.cookie,
        )

    frame = save_records(records, output)
    print(f"Saved {len(frame)} records to {output}")


if __name__ == "__main__":
    main()
