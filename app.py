import importlib
import json
import os
import threading
import traceback
import html as std_html
import dash
import pandas as pd
from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc
from flask import request as flask_request, Response
from flask_compress import Compress

from components.loaders import load_last_updated, load_acled_main, geojson_source_path

_ov = importlib.import_module("pages.1_overview")
_ac = importlib.import_module("pages.2_actor")
_ab = importlib.import_module("pages.3_about")
_al = importlib.import_module("pages.4_alerts")

PAGE_MAP = {
    "/":      _ov.layout,
    "/actor": _ac.layout,
    "/alerts": _al.layout,
    "/about": _ab.layout,
}

NAV_ITEMS = [
    {"path": "/",      "label": "Overview"},
    {"path": "/actor", "label": "Actor Analysis"},
    {"path": "/alerts", "label": "Township Alerts"},
    {"path": "/about", "label": "About"},
]

SITE_URL = "https://myanmarconflictdashboard.com"
SITE_NAME = "Myanmar Conflict Dashboard"
DEFAULT_DESCRIPTION = (
    "Township-level dashboard for exploring Myanmar conflict patterns, event trends, "
    "actor analysis, and transparent township alerts since February 2021."
)
PAGE_SEO = {
    "/": {
        "title": "Myanmar Conflict Dashboard | Township map, trends, and alerts",
        "description": (
            "Explore reported conflict patterns across Myanmar with township-level maps, "
            "event trends, actor analysis, and transparent alerting."
        ),
        "keywords": [
            "Myanmar conflict dashboard",
            "Myanmar conflict map",
            "Myanmar township conflict",
            "Myanmar conflict alerts",
            "Myanmar actor analysis",
        ],
    },
    "/actor": {
        "title": "Actor Analysis | Myanmar Conflict Dashboard",
        "description": (
            "Track one actor's geographic footprint, associated actors, and engagement pattern "
            "across Myanmar conflict events."
        ),
        "keywords": [
            "Myanmar actor analysis",
            "Myanmar conflict actors",
            "Myanmar armed conflict actor map",
        ],
    },
    "/alerts": {
        "title": "Township Alerts | Myanmar Conflict Dashboard",
        "description": (
            "See which Myanmar townships show unusually high recent armed conflict activity or "
            "reported fatalities compared with their own history."
        ),
        "keywords": [
            "Myanmar township alerts",
            "Myanmar conflict alerts",
            "Myanmar armed conflict alerts",
            "Myanmar fatality surge",
        ],
    },
    "/about": {
        "title": "About | Myanmar Conflict Dashboard",
        "description": (
            "Read about the purpose, methods, limitations, and academic context of the Myanmar Conflict Dashboard."
        ),
        "keywords": [
            "Myanmar conflict dashboard methodology",
            "Myanmar conflict dashboard about",
        ],
    },
}


def _normalize_path(path: str | None) -> str:
    path = path or "/"
    return path if path in PAGE_MAP else "/"


def _absolute_url(path: str) -> str:
    path = _normalize_path(path)
    return SITE_URL if path == "/" else f"{SITE_URL.rstrip('/')}{path}"


def _seo_for_path(path: str | None) -> dict:
    path = _normalize_path(path)
    meta = PAGE_SEO.get(path, PAGE_SEO["/"])
    return {
        "path": path,
        "title": meta["title"],
        "description": meta["description"],
        "canonical": _absolute_url(path),
        "keywords": meta.get("keywords", []),
    }


def _structured_data_json(path: str | None) -> str:
    meta = _seo_for_path(path)
    if meta["path"] == "/":
        page_type = "CollectionPage"
    elif meta["path"] == "/about":
        page_type = "AboutPage"
    else:
        page_type = "WebPage"
    modified = load_last_updated() or None

    breadcrumbs = [
        {
            "@type": "ListItem",
            "position": 1,
            "name": "Myanmar Conflict Dashboard",
            "item": SITE_URL,
        }
    ]
    if meta["path"] != "/":
        label = next((item["label"] for item in NAV_ITEMS if item["path"] == meta["path"]), meta["title"])
        breadcrumbs.append(
            {
                "@type": "ListItem",
                "position": 2,
                "name": label,
                "item": meta["canonical"],
            }
        )

    payload = [
        {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": SITE_NAME,
            "url": SITE_URL,
            "description": DEFAULT_DESCRIPTION,
            "inLanguage": "en",
            "keywords": PAGE_SEO["/"]["keywords"],
        },
        {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": breadcrumbs,
        },
        {
            "@context": "https://schema.org",
            "@type": page_type,
            "name": meta["title"],
            "url": meta["canonical"],
            "description": meta["description"],
            "isPartOf": {
                "@type": "WebSite",
                "name": SITE_NAME,
                "url": SITE_URL,
            },
            "inLanguage": "en",
            "keywords": meta.get("keywords", []),
            **({"dateModified": modified} if modified else {}),
        },
    ]
    return json.dumps(payload, ensure_ascii=False)


class SEODash(dash.Dash):
    def interpolate_index(self, **kwargs):
        try:
            path = flask_request.path or "/"
        except Exception:
            path = "/"

        meta = _seo_for_path(path)
        title = std_html.escape(meta["title"])
        description = std_html.escape(meta["description"], quote=True)
        canonical = std_html.escape(meta["canonical"], quote=True)
        structured = _structured_data_json(path)

        return f"""<!DOCTYPE html>
<html lang="en">
    <head>
        {kwargs.get("metas", "")}
        <meta name="description" content="{description}" />
        <meta name="robots" content="index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1" />
        <meta name="googlebot" content="index,follow,max-image-preview:large,max-snippet:-1,max-video-preview:-1" />
        <link rel="canonical" href="{canonical}" />
        <link rel="icon" type="image/svg+xml" href="/assets/site-icon.svg" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="{std_html.escape(SITE_NAME, quote=True)}" />
        <meta property="og:title" content="{title}" />
        <meta property="og:description" content="{description}" />
        <meta property="og:url" content="{canonical}" />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content="{title}" />
        <meta name="twitter:description" content="{description}" />
        <script type="application/ld+json">{structured}</script>
        <title>{title}</title>
        {kwargs.get("favicon", "")}
        {kwargs.get("css", "")}
    </head>
    <body>
        <noscript>
            <div style="font-family:Arial,sans-serif;max-width:840px;margin:24px auto;padding:0 16px;color:#24384d;">
                <h1>{std_html.escape(SITE_NAME)}</h1>
                <p>{std_html.escape(meta["description"])}</p>
                <p>The main sections are Overview, Actor Analysis, Township Alerts, and About.</p>
            </div>
        </noscript>
        {kwargs.get("app_entry", "")}
        <footer>
            {kwargs.get("config", "")}
            {kwargs.get("scripts", "")}
            {kwargs.get("renderer", "")}
        </footer>
    </body>
</html>"""


app = SEODash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Myanmar Conflict Dashboard",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "theme-color", "content": "#284f77"},
    ],
)
server = app.server

# Gzip / brotli compression for all responses (Plotly figure JSON compresses
# ~80%, Dash bundles ~70%). Single biggest perceived-latency win on the wire.
server.config["COMPRESS_MIMETYPES"] = [
    "text/html", "text/css", "text/xml", "text/plain",
    "application/json", "application/javascript",
    "application/geo+json",
    "image/svg+xml",
]
server.config["COMPRESS_LEVEL"] = 6
server.config["COMPRESS_MIN_SIZE"] = 500
Compress(server)

# Cache static assets (CSS, JS, fonts, icons) for a year — they're hashed by Dash.
@server.after_request
def _cache_headers(resp):
    path = (flask_request.path or "")
    # Dash bundles in /_dash-component-suites/ are content-hashed, safe to cache
    # for a year. /assets/ is hashed by Dash via ?m=mtime.
    if path.startswith("/_dash-component-suites/"):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path.startswith("/assets/"):
        resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@server.route("/geo/boundaries.geojson")
def geo_boundaries():
    # Referenced by choropleth traces as a URL (components/map_utils.py) so the
    # browser downloads the boundary geometry once instead of receiving it
    # embedded in every map figure. Served as an in-memory Response (not
    # send_from_directory) because Flask-Compress skips streamed responses for
    # gzip clients; boundaries change ~never, so a day of client cache is safe.
    src = geojson_source_path()
    resp = Response(src.read_bytes(), mimetype="application/geo+json")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    resp.set_etag(str(src.stat().st_mtime))
    return resp


@server.route("/healthz")
def healthz():
    # Verifies the data layer, not just the process: a deploy that serves an
    # error div because the parquet is missing/corrupt must fail this probe.
    try:
        latest = load_acled_main()["event_date"].max()
        if pd.isna(latest):
            raise ValueError("acled_cleaned.parquet contains no dated events")
        return Response(f"ok data-through={latest.date().isoformat()}", mimetype="text/plain")
    except Exception as e:
        return Response(f"unhealthy: {e}", status=503, mimetype="text/plain")


@server.route("/robots.txt")
def robots_txt():
    content = "\n".join([
        "User-agent: *",
        "Allow: /",
        f"Sitemap: {SITE_URL}/sitemap.xml",
        "",
    ])
    return Response(content, mimetype="text/plain")


@server.route("/sitemap.xml")
def sitemap_xml():
    lastmod = load_last_updated()
    meta = {
        "/": {"changefreq": "daily", "priority": "1.0"},
        "/actor": {"changefreq": "daily", "priority": "0.9"},
        "/alerts": {"changefreq": "daily", "priority": "0.95"},
        "/about": {"changefreq": "monthly", "priority": "0.7"},
    }
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for item in NAV_ITEMS:
        url = _absolute_url(item["path"])
        hints = meta.get(item["path"], {})
        body.append("  <url>")
        body.append(f"    <loc>{std_html.escape(url)}</loc>")
        if lastmod:
            body.append(f"    <lastmod>{std_html.escape(lastmod)}</lastmod>")
        if hints.get("changefreq"):
            body.append(f"    <changefreq>{hints['changefreq']}</changefreq>")
        if hints.get("priority"):
            body.append(f"    <priority>{hints['priority']}</priority>")
        body.append("  </url>")
    body.append("</urlset>")
    return Response("\n".join(body), mimetype="application/xml")


def topnav():
    links = [
        dcc.Link(
            item["label"],
            href=item["path"],
            id=f"mob-nav-{i}",
            className="mob-nav-link",
        )
        for i, item in enumerate(NAV_ITEMS)
    ]
    return html.Div([
        html.Div([
            html.Div("Myanmar Conflict Dashboard", className="brand-abbr"),
            html.Div("Tracking conflict since Feb 2021", className="brand-tagline"),
        ], className="topnav-brand"),
        html.Div([
            html.Span("Pages", className="topnav-nav-label"),
            html.Nav(links, className="topnav-links"),
        ], className="topnav-nav"),
    ], className="topnav-shell")


def serve_layout():
    try:
        path = flask_request.path or "/"
    except Exception:
        path = "/"

    fn = PAGE_MAP.get(path, _ov.layout)

    try:
        page_content = fn()
    except Exception as e:
        print(f"[serve_layout] ERROR rendering {path}: {e}")
        traceback.print_exc()
        page_content = html.Div(f"Error loading page: {e}",
                                style={"color": "red", "padding": "20px"})

    return dbc.Container([
        dcc.Location(id="url", refresh=False),
        html.Div(id="seo-sync", style={"display": "none"}),
        html.A("Skip to main content", href="#main-content", className="skip-link"),
        topnav(),
        html.Main(
            html.Div(page_content, id="page-content"),
            id="main-content",
            className="main",
        ),
    ], fluid=True)


app.layout = serve_layout


# ── Navigate to page ──────────────────────────────────────────────────────────
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def render_page(pathname):
    print(f"[render_page] navigating to: {pathname!r}")
    try:
        fn = PAGE_MAP.get(pathname or "/")
        if fn:
            result = fn()
            print(f"[render_page] {pathname!r} rendered OK")
            return result
        print(f"[render_page] unknown path {pathname!r}, falling back to overview")
        return _ov.layout()
    except Exception as e:
        print(f"[render_page] ERROR for {pathname!r}: {e}")
        traceback.print_exc()
        return html.Div(f"Error: {e}", style={"color": "red", "padding": "20px"})


_SEO_CLIENT_MAP = json.dumps({path: _seo_for_path(path) for path in PAGE_MAP.keys()})


app.clientside_callback(
    f"""
    function(pathname) {{
        var seoMap = {_SEO_CLIENT_MAP};
        var current = seoMap[pathname] || seoMap["/"];
        if (!current) {{
            return "";
        }}

        document.title = current.title;

        var desc = document.querySelector('meta[name="description"]');
        if (desc) desc.setAttribute("content", current.description);

        var ogTitle = document.querySelector('meta[property="og:title"]');
        if (ogTitle) ogTitle.setAttribute("content", current.title);

        var ogDesc = document.querySelector('meta[property="og:description"]');
        if (ogDesc) ogDesc.setAttribute("content", current.description);

        var ogUrl = document.querySelector('meta[property="og:url"]');
        if (ogUrl) ogUrl.setAttribute("content", current.canonical);

        var twTitle = document.querySelector('meta[name="twitter:title"]');
        if (twTitle) twTitle.setAttribute("content", current.title);

        var twDesc = document.querySelector('meta[name="twitter:description"]');
        if (twDesc) twDesc.setAttribute("content", current.description);

        var canonical = document.querySelector('link[rel="canonical"]');
        if (canonical) canonical.setAttribute("href", current.canonical);

        // Plotly's responsive mode only listens to WINDOW resizes, so charts
        // that autosize inside CSS-sized containers (one-screen Overview)
        // render at a pre-stylesheet size and never recover. Nudge them after
        // layout settles — on first load and on every page switch.
        [250, 900, 2000].forEach(function(ms) {{
            setTimeout(function() {{ window.dispatchEvent(new Event("resize")); }}, ms);
        }});

        return "";
    }}
    """,
    Output("seo-sync", "children"),
    Input("url", "pathname"),
)


# ── Highlight active nav link ─────────────────────────────────────────────────
@callback(
    [Output(f"mob-nav-{i}", "className") for i in range(len(NAV_ITEMS))],
    Input("url", "pathname"),
)
def highlight_active_nav(pathname):
    pathname = pathname or "/"
    mobile = [
        "mob-nav-link mob-nav-link--active" if item["path"] == pathname
        else "mob-nav-link"
        for item in NAV_ITEMS
    ]
    return mobile


# ── Warm data caches in the background ────────────────────────────────────────
# The alerts precompute costs ~1.3 s cold; without this the first visitor after
# every restart or data refresh pays it. Daemon thread so startup isn't blocked;
# lru_cache makes a race with an early request harmless (worst case: computed twice).
def _warm_caches():
    try:
        load_acled_main()
        _al._build_alert_frames()
        print("[warm] data caches ready")
    except Exception as e:
        print(f"[warm] cache warming failed: {e}")


threading.Thread(target=_warm_caches, daemon=True, name="cache-warm").start()


if __name__ == "__main__":
    # debug=False so the auto-reloader and unminified bundles don't slow things
    # down. Set MCD_DEBUG=1 in the environment if you need the dev reloader.
    debug = os.environ.get("MCD_DEBUG") == "1"
    app.run(debug=debug, dev_tools_ui=False, port=8050)
