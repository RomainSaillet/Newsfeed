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
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict

import anthropic
import feedparser
import requests
from dotenv import load_dotenv
from jinja2 import Environment, select_autoescape

load_dotenv()

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

def fetch_feed(feed_config: dict, category: str, max_age_hours: int) -> list[Article]:
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
            raw_summary = re.sub(r"<[^>]+>", "", raw_summary).strip()

            image_url = ""
            if hasattr(entry, "media_content") and entry.media_content:
                image_url = entry.media_content[0].get("url", "")

            article_id = hashlib.md5(entry.link.encode()).hexdigest()[:12]

            articles.append(Article(
                id=article_id,
                title=entry.get("title", "(sans titre)"),
                url=entry.get("link", ""),
                source=source,
                category=category,
                published=pub_date.isoformat(),
                summary=raw_summary,
                image_url=image_url,
            ))
        except Exception as e:
            log.debug(f"  Erreur parsing entrée {source}: {e}")

    return articles


def fetch_all_feeds(config: dict, cache: dict) -> tuple[list[Article], list[Article]]:
    """Retourne (nouveaux articles, articles du cache)."""
    settings   = config["settings"]
    new_articles    = []
    cached_articles = []

    for category, cat_data in config["categories"].items():
        log.info(f"\n[{category}]")
        for feed_cfg in cat_data["feeds"]:
            fetched = fetch_feed(feed_cfg, category, settings["max_age_hours"])
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

def translate_and_score(client: anthropic.Anthropic, articles: list[Article]) -> None:
    """Haiku — traduit et score tous les nouveaux articles en un seul appel."""
    if not articles:
        return

    lines = []
    for i, a in enumerate(articles):
        lines.append(
            f"[{i}] {a.title}\n"
            f"    {a.summary[:250] if a.summary else ''}"
        )

    prompt = f"""Veille éditoriale pour directeur de contenu. {len(articles)} articles RSS.

{chr(10).join(lines)}

Pour chaque [N] retourne un JSON :
- "i": N
- "t": titre traduit en français (naturel, percutant)
- "r": résumé 2 phrases en français
- "s": score 1-10 (importance stratégique pour un pro du contenu)

JSON uniquement, tableau, pas de markdown.
[{{"i":0,"t":"...","r":"...","s":7}}]"""

    try:
        raw  = _call(client, MODEL_CHEAP, prompt, max_tokens=6000)
        data = json.loads(_clean_json(raw))
        for item in data:
            idx = item.get("i", -1)
            if 0 <= idx < len(articles):
                articles[idx].translated_title   = item.get("t", articles[idx].title)
                articles[idx].translated_summary = item.get("r", articles[idx].summary)
                articles[idx].importance_score   = float(item.get("s", 5))
    except Exception as e:
        log.error(f"Traduction Haiku: {e}")
        for a in articles:
            a.translated_title   = a.title
            a.translated_summary = a.summary
            a.importance_score   = 5.0


def cluster_articles(client: anthropic.Anthropic, articles: list[Article]) -> list[dict]:
    """Haiku — regroupe les articles par thème."""
    if not articles:
        return []

    lines = [f"[{i}] {a.translated_title} ({a.category})" for i, a in enumerate(articles)]

    prompt = f"""Expert en stratégie éditoriale. {len(articles)} articles :

{chr(10).join(lines)}

Regroupe par THÈME éditorial (un même sujet couvert par plusieurs sources = 1 cluster).
Articles sans lien = cluster de 1.

JSON uniquement :
[{{"tf":"Titre français accrocheur","te":"english theme","idx":[0,3],"imp":8}}]"""

    try:
        raw = _call(client, MODEL_CHEAP, prompt, max_tokens=3000)
        return json.loads(_clean_json(raw))
    except Exception as e:
        log.error(f"Clustering Haiku: {e}")
        return [{"tf": a.translated_title, "te": a.category, "idx": [i], "imp": a.importance_score}
                for i, a in enumerate(articles)]


def summarize_cluster(client: anthropic.Anthropic, cluster_theme: str, articles: list[Article]) -> tuple[str, str]:
    """Sonnet — résumé éditorial + détection contradictions pour les clusters multi-sources."""
    sources = "\n".join(
        f"- [{a.source}] {a.translated_title} : {a.translated_summary}"
        for a in articles
    )

    prompt = f"""Rédacteur senior, média stratégie de contenu.

Thème : {cluster_theme}

Sources :
{sources}

Rédige :
1. Brève synthétique 3-4 phrases, ton magazine professionnel francophone.
2. Points de vue divergents / contradictions entre sources (vide si aucune).

JSON : {{"resume":"...","contradictions":"..."}}"""

    try:
        raw    = _call(client, MODEL_QUALITY, prompt, max_tokens=800)
        result = json.loads(_clean_json(raw))
        return result.get("resume", ""), result.get("contradictions", "")
    except Exception as e:
        log.warning(f"Résumé Sonnet '{cluster_theme}': {e}")
        return articles[0].translated_summary if articles else "", ""


def process_with_claude(new_articles: list[Article], cached_articles: list[Article],
                        config: dict) -> tuple[list[Cluster], list[Article]]:
    client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    settings = config["settings"]

    # 1. Haiku : traduire uniquement les nouveaux articles
    if new_articles:
        log.info(f"\nHaiku — traduction de {len(new_articles)} nouveaux articles...")
        translate_and_score(client, new_articles)

    all_articles = new_articles + cached_articles

    if not all_articles:
        return [], []

    # 2. Haiku : clustering sur tous les articles
    log.info(f"Haiku — clustering de {len(all_articles)} articles...")
    clusters_raw = cluster_articles(client, all_articles)

    # 3. Sonnet : résumés pour clusters multi-sources uniquement
    log.info("Sonnet — résumés des clusters multi-sources...")
    clusters = []

    for cr in clusters_raw:
        indices = cr.get("idx", [])
        arts    = [all_articles[i] for i in indices if 0 <= i < len(all_articles)]
        if not arts:
            continue

        cat_count = Counter(a.category for a in arts)
        dominant  = cat_count.most_common(1)[0][0]

        cluster = Cluster(
            theme=cr.get("te", ""),
            theme_fr=cr.get("tf", ""),
            articles=arts,
            importance_score=float(cr.get("imp", 5)),
            category=dominant,
        )

        if len(arts) >= settings["min_articles_for_cluster"]:
            cluster.summary_fr, cluster.contradictions = summarize_cluster(
                client, cluster.theme_fr, arts
            )
        else:
            cluster.summary_fr = arts[0].translated_summary if arts else ""

        clusters.append(cluster)

    clusters.sort(key=lambda c: c.importance_score, reverse=True)

    top_n       = settings["top_stories_count"]
    top_articles = sorted(all_articles, key=lambda a: a.importance_score, reverse=True)[:top_n]

    return clusters, top_articles


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
  <title>Veille Éditoriale — {{ generated_at }}</title>
  <style>
    :root {
      --bg: #0f172a;
      --surface: #1e293b;
      --surface2: #2d3f55;
      --border: #334155;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --yellow: #fbbf24;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Georgia', serif; background: var(--bg); color: var(--text); min-height: 100vh; }

    header { background: var(--surface); border-bottom: 2px solid var(--accent); padding: 1rem 2rem; }
    .header-top { display: flex; justify-content: space-between; align-items: baseline; }
    .site-title { font-size: 1.6rem; font-weight: bold; letter-spacing: .05em; color: var(--accent); font-family: Arial, sans-serif; }
    .update-time { font-size: .8rem; color: var(--text-muted); font-family: monospace; }

    .breaking-bar { background: var(--surface2); border-left: 4px solid var(--accent); padding: .8rem 1.5rem; margin: 1.5rem 2rem; border-radius: 0 8px 8px 0; }
    .breaking-label { font-size: .7rem; font-weight: bold; letter-spacing: .15em; color: var(--accent); font-family: monospace; margin-bottom: .6rem; }
    .top-stories { display: flex; gap: .8rem; flex-wrap: wrap; }
    .top-story-pill { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: .35rem .9rem; font-size: .83rem; display: flex; align-items: center; gap: .4rem; transition: border-color .2s; }
    .top-story-pill:hover { border-color: var(--accent); }
    .top-story-pill a { color: var(--text); text-decoration: none; }
    .importance-badge { background: var(--accent); color: var(--bg); border-radius: 10px; padding: 0 .4rem; font-size: .68rem; font-weight: bold; font-family: monospace; }

    main { max-width: 1400px; margin: 0 auto; padding: 2rem; }

    .category-section { margin-bottom: 3rem; }
    .category-header { display: flex; align-items: center; gap: .8rem; margin-bottom: 1.2rem; padding-bottom: .6rem; border-bottom: 2px solid var(--border); }
    .category-icon { font-size: 1.4rem; }
    .category-name { font-size: 1.2rem; font-weight: bold; font-family: Arial, sans-serif; }
    .category-count { font-size: .75rem; color: var(--text-muted); font-family: monospace; margin-left: auto; }

    .clusters-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(370px, 1fr)); gap: 1.2rem; }
    .cluster-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; display: flex; flex-direction: column; gap: .8rem; }
    .cluster-card.multi { border-left: 3px solid; }
    .cluster-theme { font-size: 1.05rem; font-weight: bold; font-family: Arial, sans-serif; line-height: 1.3; }
    .cluster-summary { font-size: .88rem; color: var(--text-muted); line-height: 1.65; }
    .cluster-contradictions { background: rgba(251,191,36,.1); border: 1px solid rgba(251,191,36,.4); border-radius: 6px; padding: .7rem; font-size: .82rem; color: var(--yellow); line-height: 1.5; }
    .contradictions-label { font-weight: bold; font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .3rem; }
    .cluster-sources { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: auto; }
    .source-tag { font-size: .72rem; background: var(--surface2); border-radius: 4px; padding: .15rem .5rem; color: var(--text-muted); font-family: monospace; }
    .source-tag a { color: inherit; text-decoration: none; }
    .source-tag a:hover { color: var(--accent); }
    .cluster-meta { display: flex; align-items: center; justify-content: space-between; font-size: .72rem; color: var(--text-muted); font-family: monospace; }
    .importance-bar { display: flex; gap: 2px; }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--border); }
    .dot.on { background: var(--accent); }
    .cache-badge { font-size: .65rem; color: var(--text-muted); opacity: .6; }

    footer { text-align: center; padding: 2rem; color: var(--text-muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 2rem; }

    @media (max-width: 640px) { main { padding: 1rem; } .clusters-grid { grid-template-columns: 1fr; } .breaking-bar { margin: 1rem; } }
  </style>
</head>
<body>

<header>
  <div class="header-top">
    <span class="site-title">VEILLE ÉDITORIALE</span>
    <span class="update-time">Mis à jour le {{ generated_at }}</span>
  </div>
</header>

{% if top_articles %}
<div class="breaking-bar">
  <div class="breaking-label">★ SUJETS DU MOMENT</div>
  <div class="top-stories">
    {% for a in top_articles %}
    <div class="top-story-pill">
      <span class="importance-badge">{{ a.importance_score | int }}/10</span>
      <a href="{{ a.url }}" target="_blank" rel="noopener">{{ a.translated_title | truncate(68) }}</a>
    </div>
    {% endfor %}
  </div>
</div>
{% endif %}

<main>
{% for category, cat_data in categories_with_clusters.items() %}
{% if cat_data.clusters %}
<section class="category-section">
  <div class="category-header">
    <span class="category-icon">{{ cat_data.icon }}</span>
    <span class="category-name" style="color:{{ cat_data.color }}">{{ category }}</span>
    <span class="category-count">{{ cat_data.clusters | length }} sujet(s)</span>
  </div>
  <div class="clusters-grid">
    {% for cluster in cat_data.clusters %}
    <div class="cluster-card {% if cluster.articles | length > 1 %}multi{% endif %}"
         style="{% if cluster.articles | length > 1 %}border-left-color:{{ cat_data.color }}{% endif %}">
      <div class="cluster-meta">
        <span>{{ cluster.articles | length }} source(s)</span>
        <div class="importance-bar">
          {% for i in range(10) %}<div class="dot {% if i < cluster.importance_score %}on{% endif %}"></div>{% endfor %}
        </div>
      </div>
      <div class="cluster-theme">{{ cluster.theme_fr }}</div>
      {% if cluster.summary_fr %}<div class="cluster-summary">{{ cluster.summary_fr }}</div>{% endif %}
      {% if cluster.contradictions %}
      <div class="cluster-contradictions">
        <div class="contradictions-label">⚡ Points de vue divergents</div>
        {{ cluster.contradictions }}
      </div>
      {% endif %}
      <div class="cluster-sources">
        {% for a in cluster.articles %}
        <span class="source-tag">
          <a href="{{ a.url }}" target="_blank" rel="noopener">{{ a.source }}</a>
          {% if a.from_cache %}<span class="cache-badge">↩</span>{% endif %}
        </span>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}
{% endfor %}
</main>

<footer>
  Veille Éditoriale · {{ total_articles }} articles · {{ generated_at }}
</footer>

</body>
</html>"""


def generate_html(clusters: list[Cluster], top_articles: list[Article], config: dict) -> str:
    categories_with_clusters = {}
    for cat_name, cat_data in config["categories"].items():
        cat_clusters = [c for c in clusters if c.category == cat_name]
        if cat_clusters:
            categories_with_clusters[cat_name] = {
                "clusters": cat_clusters,
                "color":    cat_data["color"],
                "icon":     cat_data["icon"],
            }
    uncategorized = [c for c in clusters if c.category not in config["categories"]]
    if uncategorized:
        categories_with_clusters["Autres"] = {"clusters": uncategorized, "color": "#6b7280", "icon": "📌"}

    env  = Environment(autoescape=select_autoescape(["html"]))
    tmpl = env.from_string(HTML_TEMPLATE)

    html = tmpl.render(
        categories_with_clusters=categories_with_clusters,
        top_articles=top_articles,
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
    clusters, top_articles = process_with_claude(new_articles, cached_articles, config)

    # Mettre à jour le cache avec les nouveaux articles traduits
    for a in new_articles:
        cache[a.id] = article_to_cache(a)
    save_cache(cache)
    log.info(f"Cache mis à jour : {len(cache)} articles")

    # Générer le dashboard
    log.info("\nGénération du dashboard...")
    output_path = generate_html(clusters, top_articles, config)

    log.info(f"\n✓ {len(clusters)} clusters | {len(top_articles)} top stories")
    log.info(f"  Ouvre : {output_path}")

    print_cost_report()


if __name__ == "__main__":
    main()
