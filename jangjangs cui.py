#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jangjang's web cui - advanced interactive version
"""

import os, sys, re, json, requests, textwrap
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from collections import Counter
from duckduckgo_search import DDGS
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box

console = Console()
APP_NAME = "jangjangs_web_cui"
CONFIG_DIR = Path.home() / f".{APP_NAME}"
CONFIG_PATH = CONFIG_DIR / "config.json"
HISTORY_PATH = CONFIG_DIR / "history.json"
DEFAULT_CONFIG = {"nickname": None, "theme": "dark", "created_at": None, "last_used": None}

def ensure_dirs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def read_config():
    ensure_dirs()
    if CONFIG_PATH.exists():
        try:
            return json.load(open(CONFIG_PATH, encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def write_config(cfg):
    ensure_dirs()
    json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def append_history(entry):
    ensure_dirs()
    data = []
    if HISTORY_PATH.exists():
        try:
            data = json.load(open(HISTORY_PATH, encoding="utf-8"))
        except Exception:
            pass
    data.insert(0, entry)
    json.dump(data[:300], open(HISTORY_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def fetch_html(url):
    r = requests.get(url, headers={"User-Agent": "jangjang-web-cui/1.0"}, timeout=15)
    r.raise_for_status()
    return r.text

def extract_text(html, selector=None):
    soup = BeautifulSoup(html, "html.parser")
    if selector:
        els = soup.select(selector)
        if not els:
            return ""
        return "\n\n".join(e.get_text(" ", strip=True) for e in els)
    return soup.get_text(" ", strip=True)

def extract_links(html, base=None):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        t = a.get_text(strip=True) or "(no text)"
        links.append((t, a["href"]))
    return links

def show_table(data, headers):
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, show_lines=False)
    for h in headers:
        table.add_column(h)
    for row in data:
        table.add_row(*map(str, row))
    console.print(table)

def sanitize_filename(s): return re.sub(r"[^\w\-_.]", "_", s)[:80] or "output"

def analyze_text(text, topn=15):
    words = re.findall(r"[A-Za-z가-힣0-9]+", text)
    common = Counter(words).most_common(topn)
    show_table(common, ["단어", "빈도"])

def summarize(text, n=5):
    lines = text.split(".")
    if len(lines) <= n: return text
    return ". ".join(lines[:n]) + "..."

def main_loop():
    cfg = read_config()
    if not cfg.get("nickname"):
        console.print("[bold green]처음 오셨네요![/bold green] 닉네임을 입력해주세요:")
        name = Prompt.ask("닉네임")
        cfg["nickname"] = name
        cfg["created_at"] = datetime.utcnow().isoformat() + "Z"
        write_config(cfg)
        console.print(f"환영합니다, [bold yellow]{name}[/bold yellow]님!")
    console.print(f"\n✨ [bold cyan]jangjang’s web cui[/bold cyan] — 환영합니다, {cfg['nickname']}님!")
    console.print("명령: fetch / links / title / meta / search / analyze / save / ls / cat / config / exit\n")

    while True:
        try:
            cmd = Prompt.ask(f"[{cfg['nickname']}]>").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n👋 종료합니다.")
            break

        if not cmd: continue
        args = cmd.split()
        c = args[0].lower()

        if c in {"exit", "quit"}:
            console.print("👋 안녕히 가세요!")
            break

        elif c == "fetch":
            if len(args) < 2:
                console.print("[red]사용법: fetch <url> [selector][/red]")
                continue
            url = args[1]
            sel = args[2] if len(args) > 2 else None
            try:
                html = fetch_html(url)
                text = extract_text(html, sel)
                console.print(textwrap.shorten(text, 800, placeholder=" ..."))
                fname = sanitize_filename(url) + ".txt"
                Path(fname).write_text(text, encoding="utf-8")
                console.print(f"[green]📁 저장됨:[/green] {fname}")
                append_history({"time": datetime.utcnow().isoformat()+"Z","url":url,"type":"fetch"})
            except Exception as e:
                console.print(f"[red]오류:[/red] {e}")

        elif c == "links":
            if len(args) < 2:
                console.print("[red]사용법: links <url>[/red]")
                continue
            try:
                html = fetch_html(args[1])
                links = extract_links(html)
                show_table([(t, h) for t, h in links[:50]], ["텍스트", "링크"])
                console.print(f"총 {len(links)}개 링크")
            except Exception as e:
                console.print(f"[red]오류:[/red] {e}")

        elif c == "title":
            if len(args) < 2:
                console.print("[red]사용법: title <url>[/red]")
                continue
            html = fetch_html(args[1])
            soup = BeautifulSoup(html, "html.parser")
            console.print(f"[bold cyan]제목:[/bold cyan] {soup.title.string if soup.title else '(없음)'}")

        elif c == "meta":
            if len(args) < 2:
                console.print("[red]사용법: meta <url>[/red]")
                continue
            html = fetch_html(args[1])
            soup = BeautifulSoup(html, "html.parser")
            metas = [(m.get("name") or m.get("property") or "(unnamed)", m.get("content","")) for m in soup.find_all("meta")]
            show_table(metas[:20], ["이름", "내용"])

        elif c == "search":
            if len(args) < 2:
                console.print("[red]사용법: search <query>[/red]")
                continue
            query = " ".join(args[1:])
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))
            show_table([(r["title"], r["href"]) for r in results], ["제목", "URL"])

        elif c == "analyze":
            if len(args) < 2:
                console.print("[red]사용법: analyze <file>[/red]")
                continue
            path = Path(args[1])
            if not path.exists():
                console.print("[red]파일 없음[/red]")
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            analyze_text(text)
            console.print("\n요약:")
            console.print(summarize(text))

        elif c == "save":
            if len(args) < 3:
                console.print("[red]사용법: save <url> <파일명>[/red]")
                continue
            html = fetch_html(args[1])
            Path(args[2]).write_text(html, encoding="utf-8")
            console.print(f"✅ 저장 완료: {args[2]}")

        elif c == "ls":
            files = os.listdir(".")
            show_table([(f,) for f in files], ["파일"])
        elif c == "cat":
            if len(args) < 2:
                console.print("[red]사용법: cat <파일>[/red]")
                continue
            try:
                text = Path(args[1]).read_text(encoding="utf-8", errors="ignore")
                console.print(textwrap.shorten(text, 1000, placeholder=" ..."))
            except Exception as e:
                console.print(f"[red]읽기 실패:[/red] {e}")

        elif c == "history":
            if HISTORY_PATH.exists():
                data = json.load(open(HISTORY_PATH, encoding="utf-8"))
                show_table([(h["time"], h["type"], h["url"]) for h in data[:15]], ["시간", "종류", "URL"])
            else:
                console.print("히스토리 없음.")
        elif c == "config":
            console.print(json.dumps(cfg, indent=2, ensure_ascii=False))
        elif c == "clear":
            os.system("cls" if os.name=="nt" else "clear")
        else:
            console.print("[yellow]명령을 인식하지 못했습니다.[/yellow]")

    cfg["last_used"] = datetime.utcnow().isoformat()+"Z"
    write_config(cfg)

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        console.print(f"[red]오류:[/red] {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jangjang's web cui PRO+ v2
- PRO에서 업글: SQLite 캐시+FTS, transformers 요약 optional, playwright 스크린샷 optional,
  robots.txt 준수, rate-limit, chunked translate, improved multi-fetch with progress.
Run: python3 jangjang_web_cui_pro_v2.py
"""

import os, sys, re, json, time, asyncio, signal, hashlib, subprocess, sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

# HTTP
import requests
import aiohttp
import aiofiles

# HTML parsing
from bs4 import BeautifulSoup

# system info
import psutil

# search
from duckduckgo_search import DDGS

# translation
from deep_translator import GoogleTranslator

# UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer
from prompt_toolkit.history import InMemoryHistory

console = Console()
session = PromptSession(history=InMemoryHistory())

APP = "jangjangs_web_cui_pro_v2"
HOME = Path.home()
CONFIG_DIR = HOME / f".{APP}"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"
CACHE_DB = CONFIG_DIR / "cache.db"
HISTORY_PATH = CONFIG_DIR / "history.json"

DEFAULT_CONFIG = {
    "nickname": None,
    "user_agent": "jangjang-web-cui-pro-v2/1.0",
    "concurrency": 6,
    "rate_limit_seconds": 0.3,
    "cache_ttl_days": 7,
    "use_transformers": False,   # enable if transformers installed and wanted
    "playwright_enabled": False
}

COMMANDS = [
  "help","fetch","multi-fetch","links","title","meta","summarize","translate",
  "search","save","cache","cache export","cache list","cache show","cache clear",
  "search-in-cache","crawl","screenshot","sysinfo","disk","mem","ps","kill","uptime",
  "ping","ls","cat","config","theme","clear","exit","history","export"
]

# ----------------- Config & DB -----------------
def read_config() -> Dict[str,Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except:
            pass
    cfg = DEFAULT_CONFIG.copy()
    cfg["created_at"] = datetime.utcnow().isoformat()+"Z"
    write_config(cfg)
    return cfg

def write_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def append_history(entry):
    data=[]
    if HISTORY_PATH.exists():
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except:
            data=[]
    data.insert(0, entry)
    HISTORY_PATH.write_text(json.dumps(data[:2000], ensure_ascii=False, indent=2), encoding="utf-8")

# SQLite cache with FTS
def init_db():
    conn = sqlite3.connect(str(CACHE_DB))
    cur = conn.cursor()
    # main table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pages (
      id INTEGER PRIMARY KEY,
      url TEXT UNIQUE,
      fetched_at TEXT,
      html TEXT,
      title TEXT,
      status INTEGER
    )""")
    # FTS for full-text (body text)
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(url, body, title, content='pages', content_rowid='id')
    """)
    conn.commit()
    return conn

def save_page_to_db(conn, url, html, status=200):
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        # remove scripts for body
        for s in soup(["script","style","noscript"]):
            s.extract()
        body = soup.get_text(" ", strip=True)
    except Exception:
        body = ""
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()+"Z"
    cur.execute("INSERT OR REPLACE INTO pages(url, fetched_at, html, title, status) VALUES (?,?,?,?,?)",
                (url, now, html, title, status))
    # upsert fts: delete existing then insert new (simple approach)
    cur.execute("SELECT id FROM pages WHERE url = ?", (url,))
    row = cur.fetchone()
    if row:
        rowid = row[0]
        cur.execute("DELETE FROM pages_fts WHERE rowid = ?", (rowid,))
        cur.execute("INSERT INTO pages_fts(rowid, url, body, title) VALUES (?,?,?,?)", (rowid, url, body, title))
    conn.commit()

def get_cached_page(conn, url, max_age_days=7):
    cur = conn.cursor()
    cur.execute("SELECT fetched_at, html, status FROM pages WHERE url = ?", (url,))
    r = cur.fetchone()
    if not r:
        return None
    fetched_at = datetime.fromisoformat(r[0].replace("Z",""))
    if datetime.utcnow() - fetched_at > timedelta(days=max_age_days):
        return None
    return {"html": r[1], "status": r[2], "fetched_at": r[0]}

def search_in_cache(conn, query, limit=25):
    cur = conn.cursor()
    cur.execute("SELECT url, title, snippet(pages_fts, 1, '...', '...', '...', 50) FROM pages_fts WHERE pages_fts MATCH ? LIMIT ?", (query, limit))
    return cur.fetchall()

# ----------------- Web helpers -----------------
def fetch_sync(url, user_agent, timeout=20):
    headers = {"User-Agent": user_agent}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text, r.status_code

async def fetch_async(session, url, user_agent, timeout=30):
    headers = {"User-Agent": user_agent}
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.text(), resp.status

def extract_text(html, selector=None, keep_whitespace=False):
    soup = BeautifulSoup(html, "html.parser")
    if selector:
        sel = soup.select(selector)
        if not sel:
            return ""
        return "\n\n".join([e.get_text(separator="\n" if keep_whitespace else " ", strip=True) for e in sel])
    for s in soup(["script","style","noscript"]): s.extract()
    return soup.get_text(separator="\n" if keep_whitespace else " ", strip=True)

def extract_links(html):
    soup = BeautifulSoup(html, "html.parser")
    out=[]
    for a in soup.find_all("a", href=True):
        out.append({"text": a.get_text(strip=True) or "(no text)", "href": a["href"]})
    return out

# ----------------- Robots & polite crawling -----------------
import urllib.robotparser
from urllib.parse import urlparse, urljoin

def allow_by_robots(url, user_agent):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(urljoin(base, "/robots.txt"))
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return True  # if robots can't be read, be permissive (but we still rate-limit)

# ----------------- Summarization (transformers optional) -----------------
USE_TRANSFORMERS = False
try:
    import transformers
    from transformers import pipeline
    # do not initialize heavy pipeline at import; lazy load on demand
    USE_TRANSFORMERS = True
except Exception:
    USE_TRANSFORMERS = False

def summarizer_fallback(text, max_sentences=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return ""
    # score by length & keyword presence (simple)
    words = re.findall(r"\w+", text.lower())
    freq = Counter(words)
    def score(s):
        sc = len(s)
        for w in re.findall(r"\w+", s.lower()):
            sc += freq.get(w,0)
        return sc
    scored = sorted(sentences, key=score, reverse=True)[:max_sentences]
    scored_sorted = sorted(scored, key=lambda s: sentences.index(s))
    return " ".join(scored_sorted)

_transformer_pipeline = None
def summarizer_transformers(text, model_name="sshleifer/distilbart-cnn-12-6", max_length=200):
    global _transformer_pipeline
    if _transformer_pipeline is None:
        console.print("[cyan]Transformers 요약 파이프라인 초기화 중... (모델 다운로드 필요)[/cyan]")
        _transformer_pipeline = pipeline("summarization", model=model_name)
    # transformers expects shorter inputs; chunk if large
    if len(text) < 1000:
        return _transformer_pipeline(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
    # chunk
    chunks = []
    start=0
    while start < len(text):
        chunk = text[start:start+8000]
        chunks.append(chunk)
        start += 8000
    sums=[]
    for c in chunks:
        out = _transformer_pipeline(c, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        sums.append(out)
    return "\n".join(sums)

def summarize_text(text, use_transformers=False):
    if use_transformers and USE_TRANSFORMERS:
        try:
            return summarizer_transformers(text)
        except Exception as e:
            console.print(f"[yellow]transformers 요약 실패, 폴백 사용: {e}[/yellow]")
            return summarizer_fallback(text)
    else:
        return summarizer_fallback(text)

# ----------------- Translation with chunking -----------------
def translate_long_text(text: str, target: str="ko"):
    max_chunk = 3000  # chars per call - safe and avoids very long requests
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    out_chunks=[]
    for c in chunks:
        try:
            t = GoogleTranslator(source="auto", target=target).translate(c)
        except Exception as e:
            t = f"[번역 실패: {e}]"
        out_chunks.append(t)
    return "\n\n".join(out_chunks)

# ----------------- Multi-fetch with progress -----------------
async def multi_fetch_urls(urls: List[str], cfg, conn=None):
    concurrency = max(1, int(cfg.get("concurrency", 4)))
    ua = cfg.get("user_agent")
    tasks=[]
    results={}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
        async with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as prog:
            task = prog.add_task("multi-fetch", total=len(urls))
            async def worker(url):
                async with sem:
                    try:
                        if not allow_by_robots(url, ua):
                            results[url] = {"status":"blocked_by_robots"}
                            prog.advance(task)
                            return
                        html, st = await fetch_async(sess, url, ua)
                        results[url] = {"status":"ok", "len": len(html)}
                        if conn:
                            save_page_to_db(conn, url, html, status=st)
                        prog.advance(task)
                    except Exception as e:
                        results[url] = {"status":"error", "error": str(e)}
                        prog.advance(task)
            for u in urls:
                tasks.append(asyncio.create_task(worker(u)))
            await asyncio.gather(*tasks)
    return results

# ----------------- Screenshot (Playwright) optional -----------------
PLAYWRIGHT_OK = False
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False

def screenshot_url(url, out_path="screenshot.png", width=1200, height=800):
    if not PLAYWRIGHT_OK:
        raise RuntimeError("Playwright 미설치: `pip install playwright` 및 `playwright install` 실행 필요")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width":width,"height":height})
        page.goto(url, timeout=30000)
        page.screenshot(path=out_path, full_page=True)
        browser.close()
    return out_path

# ----------------- System utilities -----------------
def sysinfo():
    return {"platform": sys.platform, "python": sys.version.splitlines()[0], "cpu_count": psutil.cpu_count(), "memory_gb": round(psutil.virtual_memory().total/(1024**3),2)}

def disk_usage():
    d=psutil.disk_usage("/")
    return {"total_gb":round(d.total/(1024**3),2), "used":round(d.used/(1024**3),2), "percent":d.percent}

def mem_info():
    vm=psutil.virtual_memory()
    return {"total":vm.total, "used":vm.used, "percent":vm.percent}

def list_processes(limit=40):
    procs=[]
    for p in psutil.process_iter(['pid','name','username','cpu_percent','memory_percent']):
        try:
            procs.append(p.info)
        except: pass
    return sorted(procs, key=lambda x: x.get('cpu_percent',0), reverse=True)[:limit]

def kill_pid(pid):
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False

def uptime():
    return str(datetime.now() - datetime.fromtimestamp(psutil.boot_time())).split('.')[0]

def ping_host(host, count=3):
    try:
        res = subprocess.run(["ping","-c",str(count),host], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return res.stdout or res.stderr
    except Exception as e:
        return str(e)

# ----------------- Prompt & Completion -----------------
class SimpleCompleter(Completer):
    def __init__(self, words):
        self.words = words
        self.path_completer = PathCompleter()
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text or ' ' not in text:
            for w in self.words:
                if w.startswith(text):
                    yield type("C",(object,),{"text":w})()
        else:
            # if second arg is filepath-like, delegate to path completer
            first = text.split()[0]
            if first in ("ls","cat","export","save","cache","history"):
                for c in self.path_completer.get_completions(document, complete_event):
                    yield c

# ----------------- Pretty helpers -----------------
def pretty_table(rows, headers):
    t = Table(show_header=True)
    for h in headers: t.add_column(h)
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)

def show_help():
    console.print(Panel("""주요 명령 예:
  fetch <url>
  multi-fetch <url1> <url2> ...
  links <url>
  title <url>
  meta <url>
  summarize <url|file> [--model]
  translate <url|file> <ko|en|ja>
  search <query>
  save <url> <path>
  cache list | cache show <url> | cache clear | cache export <out.db|out.json>
  search-in-cache <query>
  crawl <start_url> depth <N>
  screenshot <url> [outfile.png]
  sysinfo | disk | mem | ps | kill <pid> | uptime | ping <host>
  ls | cat <file>
  history [export <file>]
  config | clear | exit
""", title="Help"))

# ----------------- Main interactive loop -----------------
def main_loop():
    cfg = read_config()
    conn = init_db()
    if not cfg.get("nickname"):
        console.print(Panel("[bold green]처음 오셨군요! 닉네임을 입력하세요[/bold green]"))
        n = Prompt.ask("닉네임").strip() or "jangjang"
        cfg["nickname"] = n
        cfg["created_at"] = datetime.utcnow().isoformat()+"Z"
        write_config(cfg)
    name = cfg["nickname"]
    completer = SimpleCompleter(COMMANDS)
    console.print(Panel(f"[bold cyan]jangjang's web cui PRO+ v2[/bold cyan] — 환영합니다, [bold yellow]{name}[/bold yellow]님"))
    console.print("자동완성: Tab, 히스토리: ↑↓, 도움말: help\n")

    while True:
        try:
            raw = session.prompt(f"[{name}]> ", completer=completer)
        except (KeyboardInterrupt, EOFError):
            console.print("\n종료합니다.")
            break
        line = raw.strip()
        if not line: continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("exit","quit"):
            console.print("Bye!")
            break
        if cmd in ("help","?"):
            show_help(); continue
        if cmd == "clear":
            os.system("cls" if os.name=="nt" else "clear"); continue
        if cmd == "config":
            console.print(json.dumps(cfg, ensure_ascii=False, indent=2)); continue

        # fetch
        if cmd == "fetch":
            if len(parts) < 2: console.print("[red]fetch <url>[/red]"); continue
            url = parts[1]
            # robots
            if not allow_by_robots(url, cfg.get("user_agent")):
                console.print("[yellow]robots.txt에 의해 차단됨[/yellow]")
            try:
                cached = get_cached_page(conn, url, max_age_days=cfg.get("cache_ttl_days",7))
                if cached:
                    console.print(f"[green]캐시 사용: {cached.get('fetched_at')}[/green]")
                    html = cached.get("html")
                else:
                    html, st = fetch_sync(url, cfg.get("user_agent"))
                    save_page_to_db(conn, url, html, status=st)
                txt = extract_text(html)[:3000]
                console.print(Panel(txt if txt else "(본문 없음)", title=f"요약: {url}"))
                append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"fetch","url":url})
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # multi-fetch
        if cmd == "multi-fetch":
            if len(parts) < 2: console.print("[red]multi-fetch <url1> <url2>...[/red]"); continue
            urls = parts[1:]
            console.print(f"병렬 페치 {len(urls)}개 (concurrency={cfg.get('concurrency')})")
            try:
                results = asyncio.run(multi_fetch_urls(urls, cfg, conn=conn))
                rows=[]
                for u,r in results.items():
                    rows.append((u, r.get("status"), r.get("len", "-"), r.get("error","")))
                pretty_table(rows, ["URL","상태","길이","오류"])
                append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"multi-fetch","count":len(urls)})
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # links
        if cmd == "links":
            if len(parts) < 2: console.print("[red]links <url>[/red]"); continue
            url = parts[1]
            try:
                html, st = fetch_sync(url, cfg.get("user_agent"))
                links = extract_links(html)
                pretty_table([(i+1, l["text"], l["href"]) for i,l in enumerate(links[:200])], ["#","텍스트","링크"])
                append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"links","url":url,"count":len(links)})
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # title/meta
        if cmd == "title":
            if len(parts)<2: console.print("[red]title <url>[/red]"); continue
            try:
                html,st = fetch_sync(parts[1], cfg.get("user_agent"))
                soup = BeautifulSoup(html,'html.parser')
                console.print(f"[bold]제목:[/bold] {soup.title.string if soup.title else '(없음)'}")
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue
        if cmd == "meta":
            if len(parts)<2: console.print("[red]meta <url>[/red]"); continue
            try:
                html,st = fetch_sync(parts[1], cfg.get("user_agent"))
                metas = [(m.get("name") or m.get("property") or "(unnamed)", m.get("content","")) for m in BeautifulSoup(html,"html.parser").find_all("meta")]
                pretty_table(metas[:100], ["이름","내용"])
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # summarize
        if cmd == "summarize":
            if len(parts) < 2: console.print("[red]summarize <url|file> [--model][/red]"); continue
            target = parts[1]
            use_model = ("--model" in parts)
            try:
                if target.startswith("http"):
                    html,st = fetch_sync(target, cfg.get("user_agent"))
                    text = extract_text(html)
                else:
                    text = Path(target).read_text(encoding="utf-8", errors="ignore")
                s = summarize_text(text, use_transformers and cfg.get("use_transformers", False) and use_model)
                console.print(Panel(s, title=f"요약: {target}"))
                append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"summarize","target":target})
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # translate
        if cmd == "translate":
            if len(parts) < 3: console.print("[red]translate <url|file> <ko|en|ja>[/red]"); continue
            target = parts[1]; lang = parts[2]
            try:
                if target.startswith("http"):
                    html,st = fetch_sync(target, cfg.get("user_agent"))
                    text = extract_text(html)
                else:
                    text = Path(target).read_text(encoding="utf-8", errors="ignore")
                t = translate_long_text(text, target=lang)
                console.print(Panel(t[:4000], title=f"번역({lang})"))
                append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"translate","target":target,"lang":lang})
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # search (web)
        if cmd == "search":
            if len(parts) < 2: console.print("[red]search <query>[/red]"); continue
            q = " ".join(parts[1:])
            try:
                with DDGS() as ddgs:
                    res = list(ddgs.text(q, max_results=10))
                pretty_table([(r.get("title"), r.get("href")) for r in res], ["제목","URL"])
                append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"search","q":q})
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # save
        if cmd == "save":
            if len(parts) < 3: console.print("[red]save <url> <file>[/red]"); continue
            url, out = parts[1], parts[2]
            try:
                html, st = fetch_sync(url, cfg.get("user_agent"))
                Path(out).write_text(html, encoding="utf-8")
                console.print(f"[green]저장됨: {out}[/green]")
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        # cache
        if cmd == "cache":
            if len(parts) < 2: console.print("[red]cache list|show <url>|clear|export <file>[/red]"); continue
            sub = parts[1]
            if sub == "list":
                rows = []
                cur = conn.cursor()
                cur.execute("SELECT fetched_at, url FROM pages ORDER BY fetched_at DESC LIMIT 200")
                for r in cur.fetchall():
                    rows.append((r[0], r[1]))
                pretty_table(rows, ["fetched_at","url"])
            elif sub == "show" and len(parts) >= 3:
                c = get_cached_page(conn, parts[2], max_age_days=10000)
                if c:
                    console.print(Panel(c.get("html","")[:4000], title=f"캐시: {parts[2]}"))
                else:
                    console.print("[yellow]캐시 없음[/yellow]")
            elif sub == "clear":
                cur = conn.cursor()
                cur.execute("DELETE FROM pages")
                cur.execute("DELETE FROM pages_fts")
                conn.commit()
                console.print("[green]캐시 초기화 완료[/green]")
            elif sub == "export" and len(parts) >= 3:
                outp = Path(parts[2])
                # export SQLite -> copy file
                conn.commit()
                conn.close()
                import shutil
                shutil.copyfile(str(CACHE_DB), str(outp))
                conn = init_db()
                console.print(f"[green]DB 복사 완료: {outp}[/green]")
            else:
                console.print("[red]알 수 없는 cache 명령[/red]")
            continue

        # search-in-cache
        if cmd == "search-in-cache":
            if len(parts) < 2: console.print("[red]search-in-cache <query>[/red]"); continue
            q = " ".join(parts[1:])
            rows = search_in_cache(conn, q)
            pretty_table(rows, ["url","title","snippet"])
            continue

        # crawl (simple BFS with robots + depth)
        if cmd == "crawl":
            if len(parts) < 4 or parts[2] != "depth":
                console.print("[red]crawl <start_url> depth <N>[/red]"); continue
            start = parts[1]; depth = int(parts[3])
            visited=set()
            queue=[(start,0)]
            console.print(f"크롤링 시작: {start} (depth={depth})")
            while queue:
                url,d = queue.pop(0)
                if url in visited or d > depth: continue
                visited.add(url)
                try:
                    if not allow_by_robots(url, cfg.get("user_agent")):
                        console.print(f"[yellow]robots 차단: {url}[/yellow]"); continue
                    html,st = fetch_sync(url, cfg.get("user_agent"))
                    save_page_to_db(conn, url, html, status=st)
                    append_history({"time":datetime.utcnow().isoformat()+"Z","cmd":"crawl","url":url,"depth":d})
                    if d < depth:
                        for l in extract_links(html):
                            href = l["href"]
                            # normalize relative
                            try:
                                if href.startswith("http"):
                                    nxt = href
                                else:
                                    from urllib.parse import urljoin
                                    nxt = urljoin(url, href)
                                if nxt not in visited:
                                    queue.append((nxt, d+1))
                            except:
                                pass
                except Exception as e:
                    console.print(f"[red]fetch 오류 {url}: {e}[/red]")
            console.print(f"[green]크롤링 완료: 총 {len(visited)} 페이지 저장[/green]")
            continue

        # screenshot
        if cmd == "screenshot":
            if len(parts) < 2: console.print("[red]screenshot <url> [outfile.png][/red]"); continue
            out = parts[2] if len(parts) >= 3 else f"screenshot_{int(time.time())}.png"
            try:
                p = screenshot_url(parts[1], out)
                console.print(f"[green]스크린샷 저장됨: {p}[/green]")
            except Exception as e:
                console.print(f"[red]스크린샷 실패: {e}[/red]")
            continue

        # history
        if cmd == "history":
            hs = []
            if HISTORY_PATH.exists():
                try:
                    hs = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
                except:
                    hs=[]
            if len(parts) >= 3 and parts[1] == "export":
                out = Path(parts[2])
                out.write_text(json.dumps(hs, ensure_ascii=False, indent=2), encoding="utf-8")
                console.print(f"[green]내보내기 완료: {out}[/green]")
            else:
                pretty_table([(h.get("time",""), h.get("cmd",""), h.get("url","")) for h in hs[:200]], ["time","cmd","target"])
            continue

        # system commands
        if cmd == "sysinfo":
            console.print(json.dumps(sysinfo(), indent=2, ensure_ascii=False)); continue
        if cmd == "disk":
            console.print(json.dumps(disk_usage(), indent=2, ensure_ascii=False)); continue
        if cmd == "mem":
            console.print(json.dumps(mem_info(), indent=2, ensure_ascii=False)); continue
        if cmd == "ps":
            pretty_table([(p["pid"],p["name"],p.get("username",""), p.get("cpu_percent",0), round(p.get("memory_percent",0),2)) for p in list_processes(50)],
                         ["pid","name","user","cpu%","mem%"])
            continue
        if cmd == "kill":
            if len(parts) < 2: console.print("[red]kill <pid>[/red]"); continue
            try:
                pid=int(parts[1]); ok = kill_pid(pid)
                console.print("[green]종료 신호 전송[/green]" if ok else "[red]종료 실패[/red]")
            except:
                console.print("[red]유효한 PID 필요[/red]")
            continue
        if cmd == "uptime":
            console.print(f"uptime: {uptime()}"); continue
        if cmd == "ping":
            if len(parts) < 2: console.print("[red]ping <host>[/red]"); continue
            console.print(Panel(ping_host(parts[1]), title=f"ping {parts[1]}")); continue

        # file ops
        if cmd == "ls":
            files = os.listdir("."); pretty_table([(f,) for f in files], ["파일"]); continue
        if cmd == "cat":
            if len(parts) < 2: console.print("[red]cat <file>[/red]"); continue
            p = Path(parts[1])
            if not p.exists(): console.print("[red]파일 없음[/red]"); continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            console.print(Panel(txt[:4000], title=str(p))); continue

        # export text/links
        if cmd == "export":
            if len(parts) < 4: console.print("[red]export <type:text|links> <url> <outfile>[/red]"); continue
            typ, url, outfile = parts[1], parts[2], parts[3]
            try:
                html,st = fetch_sync(url, cfg.get("user_agent"))
                if typ == "text":
                    Path(outfile).write_text(extract_text(html), encoding="utf-8")
                elif typ == "links":
                    import csv
                    links = extract_links(html)
                    with open(outfile, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f); w.writerow(["text","href"])
                        for l in links: w.writerow([l["text"], l["href"]])
                console.print(f"[green]내보내기 완료: {outfile}[/green]")
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")
            continue

        console.print("[yellow]알 수 없는 명령 — help 입력[/yellow]")

    # on exit: update last_used
    cfg["last_used"] = datetime.utcnow().isoformat()+"Z"
    write_config(cfg)
    conn.close()

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        console.print(f"[red]프로그램 오류: {e}[/red]")
        raise