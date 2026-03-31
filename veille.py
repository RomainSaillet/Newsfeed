#!/usr/bin/env python3
"""
Veille Éditoriale — agrégateur RSS + IA pour dashboard Substack
Optimisé coûts : cache articles + Haiku pour traduction/clustering, Sonnet pour résumés
"""

import json
import os
import re
import sys
import time
import hashlib
import logging
import html as html_mod
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict

import anthropic
import feedparser
import requests
from dotenv import load_dotenv
from jinja2 import Environment, select_autoescape

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = BASE_DIR / "cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILE = CACHE_DIR / "articles.json"

# Modèles utilisés
MODEL_CHEAP   = "claude-haiku-4-5-20251001"   # traduction, clustering — $0.80/$4 par M tokens
MODEL_QUALITY = "claude-sonnet-4-6"            # résumés éditoriaux   — $3/$15 par M tokens

# Suivi des tokens consommés (pour estimer le coût en fin de run)
_token_usage = {"haiku_in": 0, "haiku_out": 0, "sonnet_in": 0, "sonnet_out": 0}


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class Article:
    id: str
    title: str
    url: str
    source: str
    category: str
    published: str          # ISO string pour sérialisation JSON
    summary: str = ""
    translated_title: str = ""
    translated_summary: str = ""
    image_url: str = ""
    importance_score: float = 0.0
    from_cache: bool = False


@dataclass
class Cluster:
    theme: str
    theme_fr: str
    articles: list = field(default_factory=list)
    summary_fr: str = ""
    hero_summary: str = ""
    contradictions: str = ""
    importance_score: float = 0.0
    category: str = ""


# ─── Cache ──────────────────────────────────────────────────────────────────

def load_cache(max_age_hours: int) -> dict:
    """Charge le cache et supprime les entrées trop vieilles."""
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        fresh = {k: v for k, v in data.items() if v.get("published", "") >= cutoff}
        return fresh
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def article_to_cache(a: Article) -> dict:
    return asdict(a)


def article_from_cache(d: dict) -> Article:
    a = Article(**{k: v for k, v in d.items() if k in Article.__dataclass_fields__})
    a.from_cache = True
    return a


# ─── RSS fetching ────────────────────────────────────────────────────────────

def fetch_feed(feed_config: dict, max_age_hours: int) -> list[Article]:
    url    = feed_config["url"]
    source = feed_config["name"]

    try:
        headers = {"User-Agent": "VeilleEdito/1.0 (RSS Reader)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:
        log.warning(f"  {source}: impossible de récupérer ({e})")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    articles = []

    for entry in feed.entries:
        try:
            pub_date = None
            for attr in ("published_parsed", "updated_parsed", "created_parsed"):
                if hasattr(entry, attr) and getattr(entry, attr):
                    t = getattr(entry, attr)
                    pub_date = datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
                    break
            if pub_date is None:
                pub_date = datetime.now(timezone.utc)
            if pub_date < cutoff:
                continue

            raw_summary = ""
            for attr in ("summary", "description"):
                if hasattr(entry, attr):
                    raw_summary = getattr(entry, attr)[:800]
                    break
            raw_summary = re.sub(r"<[^>]+>", " ", raw_summary)   # strip HTML tags
            raw_summary = html_mod.unescape(raw_summary)           # &amp; → & etc.
            raw_summary = re.sub(r"\s+", " ", raw_summary).strip() # espaces multiples

            image_url = ""
            # 1. media:content
            if hasattr(entry, "media_content") and entry.media_content:
                for mc in entry.media_content:
                    if mc.get("url") and (mc.get("medium") == "image" or mc.get("type", "").startswith("image") or mc.get("medium") == ""):
                        image_url = mc.get("url", "")
                        break
            # 2. media:thumbnail
            if not image_url and hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
                image_url = entry.media_thumbnail[0].get("url", "")
            # 3. enclosures
            if not image_url and hasattr(entry, "enclosures"):
                for enc in entry.enclosures:
                    if enc.get("type", "").startswith("image"):
                        image_url = enc.get("url", enc.get("href", ""))
                        break
            # 4. img tag in HTML content
            if not image_url:
                html_src = ""
                if hasattr(entry, "content") and entry.content:
                    html_src = entry.content[0].get("value", "")
                elif hasattr(entry, "summary"):
                    html_src = entry.summary or ""
                m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_src)
                if m and m.group(1).startswith("http"):
                    image_url = m.group(1)

            article_id = hashlib.md5(entry.link.encode()).hexdigest()[:12]

            articles.append(Article(
                id=article_id,
                title=entry.get("title", "(sans titre)"),
                url=entry.get("link", ""),
                source=source,
                category="",          # sera classifié par Claude selon le contenu
                published=pub_date.isoformat(),
                summary=raw_summary,
                image_url=image_url,
            ))
        except Exception as e:
            log.debug(f"  Erreur parsing entrée {source}: {e}")

    return articles


def fetch_all_feeds(config: dict, cache: dict) -> tuple[list[Article], list[Article]]:
    """Retourne (nouveaux articles, articles du cache)."""
    settings        = config["settings"]
    new_articles    = []
    cached_articles = []

    for feed_cfg in config["feeds"]:
        fetched = fetch_feed(feed_cfg, settings["max_age_hours"])
        fetched = fetched[: settings["max_articles_per_feed"]]

        n_new = n_cached = 0
        for a in fetched:
            if a.id in cache:
                cached_articles.append(article_from_cache(cache[a.id]))
                n_cached += 1
            else:
                new_articles.append(a)
                n_new += 1
        log.info(f"  {feed_cfg['name']}: {n_new} nouveaux, {n_cached} en cache")

    log.info(f"\nTotal : {len(new_articles)} nouveaux | {len(cached_articles)} depuis le cache")

    # Enrichir les nouveaux articles sans image via og:image / twitter:image
    without = [a for a in new_articles if not a.image_url and a.url]
    if without:
        log.info(f"  Recherche og:image pour {len(without)} articles sans photo...")
        ua_list = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
        ]
        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
        ]
        for a in without:
            for ua in ua_list:
                try:
                    r = requests.get(a.url, headers={"User-Agent": ua}, timeout=8, allow_redirects=True)
                    for pat in patterns:
                        m = re.search(pat, r.text)
                        if m and m.group(1).startswith("http"):
                            a.image_url = m.group(1)
                            break
                    if a.image_url:
                        break
                except Exception:
                    continue

    return new_articles, cached_articles


# ─── Claude helpers ──────────────────────────────────────────────────────────

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return raw


def _call(client: anthropic.Anthropic, model: str, prompt: str, max_tokens: int) -> str:
    global _token_usage
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    key = "haiku" if "haiku" in model else "sonnet"
    _token_usage[f"{key}_in"]  += resp.usage.input_tokens
    _token_usage[f"{key}_out"] += resp.usage.output_tokens
    return resp.content[0].text


# ─── Claude AI processing ────────────────────────────────────────────────────

def _score_and_classify_batch(client: anthropic.Anthropic, batch: list[Article],
                               categories: list[str]) -> None:
    """Haiku — score + classifie chaque article par son CONTENU (pas sa source)."""
    lines = [f"[{i}] {a.title[:180]}" for i, a in enumerate(batch)]
    cats  = ", ".join(categories)
    prompt = f"""Editorial intelligence. {len(batch)} articles (any language).
For each, based on content:
1. Score 1-10 for a French editorial director
2. Best category from: {cats}

{chr(10).join(lines)}

JSON only: [{{"i":0,"s":7,"c":"Vidéo"}}]"""
    try:
        raw  = _call(client, MODEL_CHEAP, prompt, max_tokens=1000)
        data = json.loads(_clean_json(raw))
        for item in data:
            idx = item.get("i", -1)
            if 0 <= idx < len(batch):
                batch[idx].importance_score = float(item.get("s", 5))
                cat = item.get("c", "")
                if cat in categories:
                    batch[idx].category = cat
    except Exception as e:
        log.error(f"Score/classify batch: {e}")
        for a in batch:
            a.importance_score = 5.0


def score_and_classify(client: anthropic.Anthropic, articles: list[Article],
                        categories: list[str]) -> None:
    """Haiku — score + classification par contenu (sans traduction)."""
    if not articles:
        return
    for start in range(0, len(articles), 30):
        _score_and_classify_batch(client, articles[start:start + 30], categories)


def _translate_solo_batch(client: anthropic.Anthropic, batch: list[Article]) -> None:
    """Haiku — traduit titre + résumé pour les articles en cluster solo."""
    lines = [
        f"[{i}] {a.title}\n    {a.summary[:300] if a.summary else ''}"
        for i, a in enumerate(batch)
    ]
    prompt = f"""Traduis en français ces {len(batch)} articles (anglais, chinois, etc.).

{chr(10).join(lines)}

Pour chaque [N] :
- "i": N
- "t": titre français percutant et naturel
- "r": résumé 2-3 phrases en français, informatif

JSON uniquement : [{{"i":0,"t":"...","r":"..."}}]"""
    try:
        raw  = _call(client, MODEL_CHEAP, prompt, max_tokens=3000)
        data = json.loads(_clean_json(raw))
        for item in data:
            idx = item.get("i", -1)
            if 0 <= idx < len(batch):
                batch[idx].translated_title   = item.get("t", batch[idx].title)
                batch[idx].translated_summary = item.get("r", batch[idx].summary or "")
    except Exception as e:
        log.error(f"Traduction solo batch: {e}")
        for a in batch:
            a.translated_title   = a.title
            a.translated_summary = a.summary or ""


def translate_solo_articles(client: anthropic.Anthropic, articles: list[Article]) -> None:
    """Traduit uniquement les articles des clusters solo non encore traduits (ou mal traduits)."""
    to_do = [a for a in articles if not a.translated_title or a.translated_title == a.title]
    if not to_do:
        return
    log.info(f"Haiku — traduction de {len(to_do)} articles solo...")
    for start in range(0, len(to_do), 20):
        _translate_solo_batch(client, to_do[start:start + 20])


def _cluster_batch(client: anthropic.Anthropic, batch: list[Article], offset: int) -> list[dict]:
    """Haiku — clustering d'un batch de 50 articles max (indices globaux)."""
    lines = [f"[{offset+i}] {a.title[:160]}" for i, a in enumerate(batch)]
    prompt = f"""Expert en stratégie éditoriale. {len(batch)} articles (anglais, français, chinois) :

{chr(10).join(lines)}

Regroupe par THÈME éditorial (même sujet = 1 cluster, toutes langues confondues).
Articles sans lien clair = cluster de 1.
"tf" = titre OBLIGATOIREMENT en français, accrocheur, naturel. Indices globaux (commencent à {offset}).

JSON uniquement :
[{{"tf":"Titre en français","te":"english theme","idx":[{offset}],"imp":8}}]"""
    try:
        raw = _call(client, MODEL_CHEAP, prompt, max_tokens=4000)
        return json.loads(_clean_json(raw))
    except Exception as e:
        log.error(f"Clustering batch offset={offset}: {e}")
        return [{"tf": a.title, "te": "", "idx": [offset+i], "imp": a.importance_score}
                for i, a in enumerate(batch)]


def cluster_articles(client: anthropic.Anthropic, articles: list[Article]) -> list[dict]:
    """Haiku — clustering par batches de 50, puis fusion des thèmes communs."""
    if not articles:
        return []
    BATCH = 50
    if len(articles) <= BATCH:
        return _cluster_batch(client, articles, 0)

    # Phase 1 : clusters par batch
    all_raw = []
    for start in range(0, len(articles), BATCH):
        all_raw.extend(_cluster_batch(client, articles[start:start + BATCH], start))

    # Phase 2 : fusion des thèmes redondants entre batches
    themes = [f"[{i}] {c['tf']}" for i, c in enumerate(all_raw)]
    prompt = f"""Expert éditorial. {len(all_raw)} thèmes — fusionne les doublons, garde les distincts.
Conserve TOUS les indices dans "idx".

{chr(10).join(themes)}

JSON uniquement :
[{{"tf":"Titre français","te":"english theme","idx":[0,3],"imp":8}}]"""
    try:
        raw    = _call(client, MODEL_CHEAP, prompt, max_tokens=8000)
        merged = json.loads(_clean_json(raw))
        final  = []
        for m in merged:
            real_idx = []
            for ri in m.get("idx", []):
                if 0 <= ri < len(all_raw):
                    real_idx.extend(all_raw[ri].get("idx", []))
            if real_idx:
                final.append({**m, "idx": real_idx})
        return final if final else all_raw
    except Exception as e:
        log.error(f"Fusion clustering: {e}")
        return all_raw


def summarize_cluster(client: anthropic.Anthropic, cluster_theme: str,
                      articles: list[Article]) -> tuple[str, str]:
    """Haiku — résumé éditorial + contradictions (sources en langue originale → français)."""
    sources = "\n".join(
        f"- [{a.source}] {a.title} : {a.summary[:400] if a.summary else ''}"
        for a in articles
    )
    prompt = f"""Rédacteur senior, média stratégie de contenu.
Sources en langue originale (anglais, français, chinois) — rédige TOUJOURS en français.

Thème : {cluster_theme}

Sources :
{sources}

Rédige EN FRANÇAIS :
1. Synthèse 3-4 phrases, ton magazine professionnel.
2. Points de vue divergents / contradictions (vide si aucun).

JSON : {{"resume":"...","contradictions":"..."}}"""
    try:
        raw    = _call(client, MODEL_CHEAP, prompt, max_tokens=800)
        result = json.loads(_clean_json(raw))
        return result.get("resume", ""), result.get("contradictions", "")
    except Exception as e:
        log.warning(f"Résumé '{cluster_theme}': {e}")
        return (articles[0].translated_summary or articles[0].summary) if articles else "", ""


def summarize_hero_cluster(client: anthropic.Anthropic, cluster_theme: str,
                            articles: list[Article]) -> str:
    """Sonnet — analyse longue ~1500 signes pour le hero (sources originales → français)."""
    sources = "\n".join(
        f"- [{a.source}] {a.title} : {a.summary[:500] if a.summary else ''}"
        for a in articles
    )
    prompt = f"""Rédacteur senior, grand média francophone.
Sources en langue originale — rédige EN FRANÇAIS uniquement.

Thème : {cluster_theme}

Sources :
{sources}

Rédige une analyse de 1400 à 1500 signes : synthèse des sources, contexte, enjeux, perspectives.
Ton magazine professionnel, fluide. Texte uniquement, sans titre ni JSON."""
    try:
        return _call(client, MODEL_QUALITY, prompt, max_tokens=700).strip()
    except Exception as e:
        log.warning(f"Hero summary '{cluster_theme}': {e}")
        return ""


def process_with_claude(new_articles: list[Article], cached_articles: list[Article],
                        config: dict) -> tuple[list[Cluster], list[Cluster]]:
    client     = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    settings   = config["settings"]
    categories = list(config["categories"].keys())

    # 1. Haiku : score + classification par contenu (pas de traduction)
    if new_articles:
        log.info(f"\nHaiku — scoring/classification de {len(new_articles)} nouveaux articles...")
        score_and_classify(client, new_articles, categories)

    # Classifier aussi les articles en cache sans catégorie valide
    uncategorized_cached = [a for a in cached_articles if a.category not in categories]
    if uncategorized_cached:
        log.info(f"Haiku — classification de {len(uncategorized_cached)} articles en cache sans catégorie...")
        score_and_classify(client, uncategorized_cached, categories)
    # Fallback si toujours sans catégorie après classification
    for a in cached_articles:
        if a.category not in categories:
            a.category = categories[0]

    all_articles = new_articles + cached_articles
    if not all_articles:
        return [], []

    # 2. Haiku : clustering multilingue par batches
    log.info(f"Haiku — clustering de {len(all_articles)} articles...")
    clusters_raw = cluster_articles(client, all_articles)

    # 3. Construction des clusters + résumés Sonnet
    log.info("Haiku — résumés des clusters multi-sources...")
    clusters     = []
    solo_articles = []

    for cr in clusters_raw:
        indices = cr.get("idx", [])
        arts    = [all_articles[i] for i in indices if 0 <= i < len(all_articles)]
        if not arts:
            continue

        # Catégorie dominante parmi les articles du cluster
        cat_count = Counter(a.category for a in arts if a.category)
        dominant  = cat_count.most_common(1)[0][0] if cat_count else categories[0]

        cluster = Cluster(
            theme=cr.get("te", ""),
            theme_fr=cr.get("tf", ""),
            articles=arts,
            importance_score=float(cr.get("imp", 5)),
            category=dominant,
        )

        if len(arts) >= settings["min_articles_for_cluster"]:
            # Multi-sources : résumé Sonnet en français depuis les originaux
            cluster.summary_fr, cluster.contradictions = summarize_cluster(
                client, cluster.theme_fr, arts
            )
        else:
            # Solo : traduction différée
            solo_articles.extend(arts)

        clusters.append(cluster)

    # 4. Haiku : traduction sélective des articles solo non encore traduits
    translate_solo_articles(client, solo_articles)
    for cluster in clusters:
        if not cluster.summary_fr and cluster.articles:
            a = cluster.articles[0]
            # theme_fr vient du clustering (déjà en français) — on ne l'écrase pas
            cluster.summary_fr = a.translated_summary or a.summary or ""

    clusters.sort(key=lambda c: c.importance_score, reverse=True)

    # 5. Top clusters pour le hero + résumés longs
    top_n        = settings.get("top_stories_count", 10)
    top_clusters = clusters[:top_n]
    hero_count   = min(5, len(top_clusters))
    log.info(f"Sonnet — résumés hero pour {hero_count} top clusters...")
    for cluster in top_clusters[:hero_count]:
        cluster.hero_summary = summarize_hero_cluster(client, cluster.theme_fr, cluster.articles)

    return clusters, top_clusters


# ─── Cost report ─────────────────────────────────────────────────────────────

def print_cost_report() -> None:
    u = _token_usage
    haiku_cost  = (u["haiku_in"]  / 1_000_000 * 0.80) + (u["haiku_out"]  / 1_000_000 * 4.00)
    sonnet_cost = (u["sonnet_in"] / 1_000_000 * 3.00) + (u["sonnet_out"] / 1_000_000 * 15.00)
    total = haiku_cost + sonnet_cost

    log.info("\n── Coût estimé de ce run ──────────────────")
    log.info(f"  Haiku  : {u['haiku_in']:>6} in / {u['haiku_out']:>5} out → ${haiku_cost:.4f}")
    log.info(f"  Sonnet : {u['sonnet_in']:>6} in / {u['sonnet_out']:>5} out → ${sonnet_cost:.4f}")
    log.info(f"  TOTAL  : ${total:.4f}  (~{total * 0.92:.4f} €)")
    log.info("────────────────────────────────────────────")


# ─── HTML generation ─────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Veille Éditoriale</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:     #f5f5f7;
      --card:   #ffffff;
      --text:   #1d1d1f;
      --text-2: #6e6e73;
      --text-3: #86868b;
      --border: rgba(0,0,0,0.08);
      --accent: #0071e3;
      --radius: 16px;
      --card-w: 280px;
      --card-h: 400px;
      --font:   -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    }
    body {
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    /* ── Header ── */
    header {
      position: sticky; top: 0; z-index: 100;
      background: rgba(255,255,255,0.85);
      backdrop-filter: saturate(180%) blur(20px);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
      border-bottom: 1px solid var(--border);
    }
    .header-inner {
      max-width: 1400px; margin: 0 auto; padding: 0 2rem;
      height: 52px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { font-size: 1rem; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
    .header-date { font-size: 0.75rem; color: var(--text-3); }

    /* ── Nav ── */
    nav {
      background: rgba(255,255,255,0.85);
      backdrop-filter: saturate(180%) blur(20px);
      -webkit-backdrop-filter: saturate(180%) blur(20px);
      border-bottom: 1px solid var(--border);
    }
    .nav-inner {
      max-width: 1400px; margin: 0 auto; padding: 0 2rem;
      display: flex; overflow-x: auto; scrollbar-width: none;
    }
    .nav-inner::-webkit-scrollbar { display: none; }
    .nav-tab {
      padding: 0.75rem 1.2rem; font-size: 0.82rem; font-weight: 500;
      color: var(--text-2); text-decoration: none;
      border-bottom: 2px solid transparent; white-space: nowrap;
      transition: color 0.15s, border-color 0.15s;
    }
    .nav-tab:hover { color: var(--text); }
    .nav-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

    /* ── Hero ── */
    .hero {
      background: #1d1d1f;
      padding: 2.5rem 2rem 3rem;
    }
    .hero-inner { max-width: 1400px; margin: 0 auto; }
    .hero-label {
      font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
      text-transform: uppercase; color: rgba(255,255,255,0.35);
      margin-bottom: 1.25rem;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto;
      gap: 0.75rem;
    }
    /* Conteneur flip (perspective) */
    .hero-flip-wrap {
      position: relative; border-radius: 14px;
      overflow: hidden; cursor: pointer; background: #2a2a2d;
    }
    .hero-flip-wrap.hero-card--featured { grid-row: span 2; min-height: 460px; }
    .hero-flip-wrap.hero-card--small { min-height: 320px; }
    /* Élément 3D rotatif */
    .hero-flip-inner {
      width: 100%; height: 100%; min-height: inherit;
      position: relative;
      transform-style: preserve-3d;
      perspective: 1200px;
      transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .hero-flip-wrap.flipped .hero-flip-inner { transform: rotateY(180deg); }
    /* Faces avant / arrière */
    .hero-card-front, .hero-card-back {
      position: absolute; inset: 0; border-radius: 14px;
      backface-visibility: hidden; -webkit-backface-visibility: hidden;
      display: flex; flex-direction: column; justify-content: flex-end;
    }
    .hero-card-back {
      transform: rotateY(180deg);
      background: #141417;
      padding: 1.75rem 2rem;
      justify-content: flex-start;
      overflow-y: auto;
    }
    /* Éléments face avant */
    .hero-card-bg {
      position: absolute; inset: 0; border-radius: 14px;
      background-size: cover; background-position: center;
    }
    .hero-card-overlay {
      position: absolute; inset: 0; border-radius: 14px;
      background: linear-gradient(to top, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.3) 55%, rgba(0,0,0,0.05) 100%);
    }
    .hero-card-body { position: relative; z-index: 1; padding: 1.25rem 1.35rem 1.4rem; }
    .hero-card-eyebrow {
      font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
      text-transform: uppercase; color: rgba(255,255,255,0.55);
      margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .hero-card-score {
      background: rgba(255,255,255,0.12); border-radius: 20px;
      padding: 0.1rem 0.45rem; font-size: 0.6rem; color: rgba(255,255,255,0.6);
    }
    .hero-card--featured .hero-card-title {
      font-size: 1.35rem; font-weight: 700; line-height: 1.3;
      letter-spacing: -0.025em; color: #ffffff;
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    }
    .hero-card--small .hero-card-title {
      font-size: 0.92rem; font-weight: 600; line-height: 1.35;
      letter-spacing: -0.015em; color: #ffffff;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .hero-flip-hint { font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 0.55rem; }
    /* Éléments face arrière */
    .hero-back-header {
      font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em;
      text-transform: uppercase; color: rgba(255,255,255,0.35); margin-bottom: 0.5rem;
    }
    .hero-back-title {
      font-size: 1rem; font-weight: 700; color: #fff;
      line-height: 1.3; margin-bottom: 1rem;
    }
    .hero-back-summary {
      font-size: 0.82rem; color: rgba(255,255,255,0.72);
      line-height: 1.65; flex: 1; overflow-y: auto; margin-bottom: 1rem;
    }
    .hero-back-sources { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.8rem; }
    .source-chip {
      font-size: 0.65rem; background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; padding: 0.2rem 0.6rem;
    }
    .source-chip a { color: rgba(255,255,255,0.55); text-decoration: none; }
    .source-chip a:hover { color: #fff; }
    .hero-back-cta {
      display: inline-block; font-size: 0.75rem; font-weight: 600;
      color: #fff; text-decoration: none;
      background: rgba(255,255,255,0.1); border-radius: 8px; padding: 0.5rem 1rem;
      transition: background 0.15s;
    }
    .hero-back-cta:hover { background: rgba(255,255,255,0.2); }

    @media (max-width: 700px) {
      .hero-grid { grid-template-columns: 1fr; }
      .hero-flip-wrap.hero-card--featured { min-height: 300px; grid-row: span 1; }
      .hero-flip-wrap.hero-card--small { min-height: 220px; }
    }

    /* ── Main ── */
    main { max-width: 1400px; margin: 0 auto; padding: 3rem 0 5rem; }

    /* ── Section ── */
    .section { margin-bottom: 4rem; }
    .section-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 2rem; margin-bottom: 1.25rem;
    }
    .section-title-wrap { display: flex; align-items: center; gap: 0.6rem; }
    .section-icon { font-size: 1.1rem; }
    .section-title { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.03em; }
    .section-count { font-size: 0.75rem; color: var(--text-3); }
    .section-arrows { display: flex; gap: 0.4rem; }
    .arrow-btn {
      width: 32px; height: 32px; border-radius: 50%;
      background: var(--card); border: 1px solid var(--border);
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; font-size: 0.9rem; color: var(--text-2);
      transition: background 0.15s, color 0.15s;
    }
    .arrow-btn:hover { background: var(--accent); color: white; border-color: var(--accent); }

    /* ── Cards row (horizontal scroll) ── */
    .cards-row {
      display: flex; gap: 1rem;
      overflow-x: auto;
      scroll-snap-type: x mandatory;
      scroll-behavior: smooth;
      scrollbar-width: none;
      padding: 0.5rem 2rem 1.5rem;
      -webkit-overflow-scrolling: touch;
    }
    .cards-row::-webkit-scrollbar { display: none; }

    /* ── Flip card ── */
    .flip-wrap {
      flex: 0 0 var(--card-w);
      height: var(--card-h);
      perspective: 1200px;
      cursor: pointer;
      scroll-snap-align: start;
    }
    .flip-wrap.wide { flex: 0 0 calc(var(--card-w) * 1.45); }

    .flip-inner {
      width: 100%; height: 100%;
      position: relative;
      transform-style: preserve-3d;
      transition: transform 0.55s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .flip-wrap.flipped .flip-inner { transform: rotateY(180deg); }

    .card-front, .card-back {
      position: absolute; inset: 0;
      backface-visibility: hidden;
      -webkit-backface-visibility: hidden;
      border-radius: var(--radius);
      overflow: hidden;
    }
    .card-front {
      background: var(--card);
      display: flex; flex-direction: column;
      box-shadow: 0 2px 16px rgba(0,0,0,0.07);
      transition: box-shadow 0.2s;
    }
    .flip-wrap:hover .card-front { box-shadow: 0 8px 32px rgba(0,0,0,0.13); }
    .card-back {
      transform: rotateY(180deg);
      background: var(--card);
      display: flex; flex-direction: column;
      box-shadow: 0 8px 32px rgba(0,0,0,0.13);
      padding: 1.25rem;
    }

    /* ── Front face ── */
    .card-img-wrap {
      width: 100%; flex: 0 0 55%; overflow: hidden; position: relative;
    }
    .flip-wrap.wide .card-img-wrap { flex: 0 0 60%; }
    .card-img {
      width: 100%; height: 100%; object-fit: cover;
      transition: transform 0.4s ease;
    }
    .flip-wrap:hover .card-img { transform: scale(1.04); }
    .card-placeholder {
      width: 100%; height: 100%;
      display: flex; align-items: center; justify-content: center;
      font-size: 2.4rem; opacity: 0.6;
    }
    .card-front-body {
      flex: 1; padding: 0.9rem 1rem 1rem;
      display: flex; flex-direction: column; gap: 0.3rem; overflow: hidden;
    }
    .card-eyebrow {
      font-size: 0.65rem; font-weight: 600; letter-spacing: 0.05em;
      text-transform: uppercase; color: var(--accent);
    }
    .card-title {
      font-size: 0.92rem; font-weight: 600; line-height: 1.35;
      letter-spacing: -0.015em; color: var(--text);
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    }
    .flip-wrap.wide .card-title { font-size: 1.1rem; -webkit-line-clamp: 4; }
    .card-flip-hint {
      margin-top: auto; font-size: 0.65rem; color: var(--text-3);
    }

    /* ── Back face ── */
    .card-back-header {
      font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
      text-transform: uppercase; color: var(--accent); margin-bottom: 0.4rem;
    }
    .card-back-title {
      font-size: 0.88rem; font-weight: 600; line-height: 1.3;
      letter-spacing: -0.01em; margin-bottom: 0.6rem;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .card-back-summary {
      font-size: 0.79rem; color: var(--text-2); line-height: 1.6;
      flex: 1; overflow: hidden;
      display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical;
    }
    .card-sources {
      display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.7rem;
    }
    .source-chip {
      font-size: 0.65rem; font-weight: 500; background: var(--bg);
      border-radius: 20px; padding: 0.18rem 0.55rem; color: var(--text-2);
    }
    .source-chip a { color: inherit; text-decoration: none; }
    .source-chip a:hover { color: var(--accent); }
    .card-back-cta {
      margin-top: 0.65rem;
      font-size: 0.75rem; font-weight: 600; color: var(--accent);
      text-decoration: none; display: inline-flex; align-items: center; gap: 0.2rem;
    }
    .card-back-cta:hover { text-decoration: underline; }

    .contradiction {
      margin-top: 0.6rem; background: #fff8ed;
      border-left: 2px solid #f5a623; border-radius: 0 6px 6px 0;
      padding: 0.45rem 0.6rem;
    }
    .contradiction-label {
      font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; color: #c17f00; margin-bottom: 0.15rem;
    }
    .contradiction-text {
      font-size: 0.72rem; color: #7a5200; line-height: 1.5;
      display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    }

    /* ── Footer ── */
    footer {
      border-top: 1px solid var(--border); padding: 2rem;
      text-align: center; color: var(--text-3); font-size: 0.75rem;
      max-width: 1400px; margin: 0 auto;
    }

    @media (max-width: 600px) {
      :root { --card-w: 240px; --card-h: 370px; }
      .section-header, .cards-row { padding-left: 1rem; padding-right: 1rem; }
      .header-inner, .nav-inner { padding: 0 1rem; }
      .hero { padding: 1.5rem 1rem 2rem; }
      main { padding-top: 2rem; }
    }
  </style>
</head>
<body>

<header>
  <div class="header-inner">
    <span class="logo">Veille Éditoriale</span>
    <span class="header-date">{{ generated_at }}</span>
  </div>
</header>

<nav>
  <div class="nav-inner">
    <a href="#top" class="nav-tab active">Tout</a>
    {% for category in categories_with_clusters.keys() %}
    <a href="#{{ category | slugify }}" class="nav-tab">{{ category }}</a>
    {% endfor %}
  </div>
</nav>

{% if top_clusters %}
<div class="hero">
  <div class="hero-inner">
    <div class="hero-label">À la une</div>
    <div class="hero-grid">
      {% for cluster in top_clusters[:5] %}
      {% set best_img = namespace(url='') %}
      {% for a in cluster.articles %}{% if a.image_url and not best_img.url %}{% set best_img.url = a.image_url %}{% endif %}{% endfor %}
      <div class="hero-flip-wrap {% if loop.first %}hero-card--featured{% else %}hero-card--small{% endif %}" onclick="flipHeroCard(this)">
        <div class="hero-flip-inner">
          <!-- FRONT -->
          <div class="hero-card-front">
            {% if best_img.url %}
            <div class="hero-card-bg" style="background-image:url('{{ best_img.url }}')"></div>
            {% endif %}
            <div class="hero-card-overlay"></div>
            <div class="hero-card-body">
              <div class="hero-card-eyebrow">
                {% if cluster.articles | length == 1 %}{{ cluster.articles[0].source }}{% else %}{{ cluster.articles | length }} sources{% endif %}
                <span class="hero-card-score">{{ cluster.importance_score | int }}/10</span>
              </div>
              <div class="hero-card-title">{% if cluster.articles | length == 1 and cluster.articles[0].translated_title %}{{ cluster.articles[0].translated_title }}{% else %}{{ cluster.theme_fr }}{% endif %}</div>
              <div class="hero-flip-hint">Appuyer pour résumé &#8594;</div>
            </div>
          </div>
          <!-- BACK -->
          <div class="hero-card-back">
            <div class="hero-back-header">Analyse</div>
            <div class="hero-back-title">{% if cluster.articles | length == 1 and cluster.articles[0].translated_title %}{{ cluster.articles[0].translated_title }}{% else %}{{ cluster.theme_fr }}{% endif %}</div>
            <div class="hero-back-summary">{{ cluster.hero_summary if cluster.hero_summary else cluster.summary_fr }}</div>
            <div class="hero-back-sources">
              {% for a in cluster.articles %}
              <span class="source-chip"><a href="{{ a.url }}" target="_blank" rel="noopener" onclick="event.stopPropagation()">{{ a.source }}</a></span>
              {% endfor %}
            </div>
            <a class="hero-back-cta" href="{{ cluster.articles[0].url }}" target="_blank" rel="noopener" onclick="event.stopPropagation()">
              Lire l'article complet &#8599;
            </a>
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
</div>
{% endif %}

<main id="top">
{% for category, cat_data in categories_with_clusters.items() %}
{% if cat_data.clusters %}
<section class="section" id="{{ category | slugify }}">
  <div class="section-header">
    <div class="section-title-wrap">
      <span class="section-icon">{{ cat_data.icon }}</span>
      <h2 class="section-title" style="color:{{ cat_data.color }}">{{ category }}</h2>
    </div>
    <div style="display:flex;align-items:center;gap:1rem">
      <span class="section-count">{{ cat_data.clusters | length }} article{% if cat_data.clusters | length > 1 %}s{% endif %}</span>
      <div class="section-arrows">
        <button class="arrow-btn" onclick="scrollRow('{{ category | slugify }}', -1)" aria-label="Précédent">&#8592;</button>
        <button class="arrow-btn" onclick="scrollRow('{{ category | slugify }}', 1)" aria-label="Suivant">&#8594;</button>
      </div>
    </div>
  </div>
  <div class="cards-row" id="row-{{ category | slugify }}">
    {% for cluster in cat_data.clusters %}
    {% set best_img = namespace(url='') %}
    {% for a in cluster.articles %}{% if a.image_url and not best_img.url %}{% set best_img.url = a.image_url %}{% endif %}{% endfor %}
    <div class="flip-wrap {% if loop.first %}wide{% endif %}" onclick="flipCard(this)">
      <div class="flip-inner">
        <!-- FRONT -->
        <div class="card-front">
          <div class="card-img-wrap">
            {% if best_img.url %}
            <img class="card-img" src="{{ best_img.url }}" alt="" loading="lazy"
                 onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
            <div class="card-placeholder" style="background:{{ cat_data.gradient }};display:none">{{ cat_data.icon }}</div>
            {% else %}
            <div class="card-placeholder" style="background:{{ cat_data.gradient }}">{{ cat_data.icon }}</div>
            {% endif %}
          </div>
          <div class="card-front-body">
            <div class="card-eyebrow">
              {% if cluster.articles | length == 1 %}{{ cluster.articles[0].source }}{% else %}{{ cluster.articles | length }} sources{% endif %}
              &nbsp;·&nbsp;{{ cluster.articles[0].published | timeago }}
            </div>
            <h3 class="card-title">{% if cluster.articles | length == 1 and cluster.articles[0].translated_title %}{{ cluster.articles[0].translated_title }}{% else %}{{ cluster.theme_fr }}{% endif %}</h3>
            <div class="card-flip-hint">Appuyer pour résumé &#8594;</div>
          </div>
        </div>
        <!-- BACK -->
        <div class="card-back">
          <div class="card-back-header">Résumé</div>
          <div class="card-back-title">{% if cluster.articles | length == 1 and cluster.articles[0].translated_title %}{{ cluster.articles[0].translated_title }}{% else %}{{ cluster.theme_fr }}{% endif %}</div>
          <p class="card-back-summary">{{ cluster.summary_fr if cluster.summary_fr else (cluster.articles[0].translated_summary if cluster.articles[0].translated_summary else cluster.articles[0].summary) }}</p>
          <div class="card-sources">
            {% for a in cluster.articles %}
            <span class="source-chip"><a href="{{ a.url }}" target="_blank" rel="noopener" onclick="event.stopPropagation()">{{ a.source }}</a></span>
            {% endfor %}
          </div>
          {% if cluster.contradictions %}
          <div class="contradiction">
            <div class="contradiction-label">Points de vue divergents</div>
            <p class="contradiction-text">{{ cluster.contradictions }}</p>
          </div>
          {% endif %}
          <a class="card-back-cta" href="{{ cluster.articles[0].url }}" target="_blank" rel="noopener" onclick="event.stopPropagation()">
            Lire l'article complet &#8599;
          </a>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}
{% endfor %}
</main>

<footer>Veille Éditoriale &mdash; {{ total_articles }} articles analysés &mdash; {{ generated_at }}</footer>

<script>
function flipCard(el) {
  const row = el.closest('.cards-row');
  if (row) {
    row.querySelectorAll('.flip-wrap.flipped').forEach(function(c) {
      if (c !== el) c.classList.remove('flipped');
    });
  }
  el.classList.toggle('flipped');
}
function flipHeroCard(el) {
  document.querySelectorAll('.hero-flip-wrap.flipped').forEach(function(c) {
    if (c !== el) c.classList.remove('flipped');
  });
  el.classList.toggle('flipped');
}
function scrollRow(id, dir) {
  var row = document.getElementById('row-' + id);
  if (!row) return;
  var cardW = (row.querySelector('.flip-wrap') || {offsetWidth: 290}).offsetWidth;
  row.scrollBy({ left: dir * (cardW + 16) * 3, behavior: 'smooth' });
}
</script>
</body>
</html>"""


def _timeago(iso_str: str) -> str:
    try:
        pub  = datetime.fromisoformat(iso_str)
        diff = datetime.now(timezone.utc) - pub
        h    = int(diff.total_seconds() / 3600)
        if h < 1:   return "à l'instant"
        if h < 24:  return f"il y a {h}h"
        if h < 48:  return "hier"
        return f"il y a {h // 24}j"
    except Exception:
        return ""


def _slugify(s: str) -> str:
    replacements = {"é":"e","è":"e","ê":"e","ë":"e","à":"a","â":"a","î":"i","ï":"i","ô":"o","ù":"u","û":"u","ç":"c"," ":"-"}
    out = s.lower()
    for k, v in replacements.items():
        out = out.replace(k, v)
    return re.sub(r"[^a-z0-9\-]", "", out)


def generate_html(clusters: list[Cluster], top_clusters: list[Cluster], config: dict) -> str:
    categories_with_clusters = {}
    top_n = config.get("settings", {}).get("top_stories_count", 10)
    for cat_name, cat_data in config["categories"].items():
        cat_clusters = [c for c in clusters if c.category == cat_name]
        cat_clusters = sorted(cat_clusters, key=lambda c: c.importance_score, reverse=True)[:top_n]
        if cat_clusters:
            categories_with_clusters[cat_name] = {
                "clusters":  cat_clusters,
                "color":     cat_data["color"],
                "icon":      cat_data["icon"],
                "gradient":  cat_data.get("gradient", "linear-gradient(135deg,#1a1a2e,#16213e)"),
            }
    uncategorized = [c for c in clusters if c.category not in config["categories"]]
    if uncategorized:
        categories_with_clusters["Autres"] = {
            "clusters": uncategorized, "color": "#6b7280", "icon": "📌",
            "gradient": "linear-gradient(135deg,#2d2d2d,#1a1a1a)",
        }

    env = Environment(autoescape=select_autoescape(["html"]))
    env.filters["timeago"] = _timeago
    env.filters["slugify"] = _slugify
    tmpl = env.from_string(HTML_TEMPLATE)

    html = tmpl.render(
        categories_with_clusters=categories_with_clusters,
        top_clusters=top_clusters,
        generated_at=datetime.now().strftime("%d/%m/%Y à %Hh%M"),
        total_articles=sum(len(c.articles) for c in clusters),
    )

    out = OUTPUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    log.info(f"Dashboard → {out}")
    return str(out)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY manquante. Copie .env.example en .env et remplis ta clé.")
        sys.exit(1)

    config_path = BASE_DIR / "feeds.json"
    if not config_path.exists():
        log.error("feeds.json introuvable.")
        sys.exit(1)

    config   = json.loads(config_path.read_text())
    settings = config["settings"]

    log.info("=== Veille Éditoriale ===")

    # Charger le cache
    cache = load_cache(settings["max_age_hours"])
    log.info(f"Cache : {len(cache)} articles")

    # Fetch RSS
    log.info(f"\nRécupération des flux ({settings['max_age_hours']}h max)...")
    new_articles, cached_articles = fetch_all_feeds(config, cache)

    if not new_articles and not cached_articles:
        log.warning("Aucun article. Vérifie ta connexion ou les URLs des flux.")
        sys.exit(0)

    # Traitement IA
    log.info(f"\nTraitement IA...")
    clusters, top_clusters = process_with_claude(new_articles, cached_articles, config)

    # Mettre à jour le cache : nouveaux articles + articles re-classifiés
    for a in new_articles:
        cache[a.id] = article_to_cache(a)
    for a in cached_articles:
        if a.id in cache:
            cache[a.id]["category"] = a.category  # persist la catégorie mise à jour
    save_cache(cache)
    log.info(f"Cache mis à jour : {len(cache)} articles")

    # Générer le dashboard
    log.info("\nGénération du dashboard...")
    output_path = generate_html(clusters, top_clusters, config)

    log.info(f"\n✓ {len(clusters)} clusters | {len(top_clusters)} top clusters")
    log.info(f"  Ouvre : {output_path}")

    print_cost_report()


if __name__ == "__main__":
    main()
