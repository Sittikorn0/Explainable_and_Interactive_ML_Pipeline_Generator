"""ml_process/features/export.py — CSV and HTML export utilities"""
import datetime
import numpy as np
import pandas as pd


# ── CSV builders ──────────────────────────────────────────────────────────────

def build_leaderboard_df(competition: dict) -> pd.DataFrame:
    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"], reverse=True,
    )
    errors = [(k, v) for k, v in competition.items() if v["cv_score"] is None]
    rows = []
    for i, (_, res) in enumerate(ranked):
        params = " | ".join(f"{k}={v}" for k, v in res["best_params"].items()) if res["best_params"] else "—"
        rows.append({"Rank": i + 1, "Model": res["label"],
                     "CV Score": res["cv_score"], "±Std": res["cv_std"], "Best Params": params})
    for _, res in errors:
        rows.append({"Rank": "—", "Model": res["label"],
                     "CV Score": None, "±Std": None, "Best Params": res.get("error", "")})
    return pd.DataFrame(rows)


def build_predictions_df(y_test, y_pred, task_type: str) -> pd.DataFrame:
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    if task_type == "classification":
        df["Correct"] = (y_test == y_pred)
    else:
        df["Error"] = y_pred - y_test
        df["Abs Error"] = np.abs(y_pred - y_test)
    return df


# ── HTML fragment builders ────────────────────────────────────────────────────

def _leaderboard_html(competition: dict, best_label: str) -> str:
    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"], reverse=True,
    )
    errors = [(k, v) for k, v in competition.items() if v["cv_score"] is None]
    rows = ""
    for i, (_, res) in enumerate(ranked):
        is_best = res["label"] == best_label
        params  = " · ".join(f"{k}={v}" for k, v in res["best_params"].items()) if res["best_params"] else "—"
        rank_cls = "rank-top" if i < 3 else "rank-num"
        best_tag = "<span class='best-tag'>best</span>" if is_best else ""
        row_cls  = "row-best" if is_best else ""
        rows += (
            f'<tr class="{row_cls}">'
            f'<td><span class="{rank_cls}">{i+1}</span></td>'
            f'<td><span class="model-name">{res["label"]}</span>{best_tag}</td>'
            f'<td class="num">{res["cv_score"]:.4f}</td>'
            f'<td class="num muted">±{res["cv_std"]:.4f}</td>'
            f'<td class="params-cell">{params}</td>'
            f'</tr>'
        )
    for _, res in errors:
        rows += (
            f'<tr class="row-error">'
            f'<td><span class="rank-num">—</span></td>'
            f'<td class="model-name">{res["label"]}</td>'
            f'<td colspan="3" class="error-msg">{res.get("error","")}</td>'
            f'</tr>'
        )
    return (
        f'<table class="tbl">'
        f'<thead><tr><th style="width:48px">Rank</th><th>Model</th>'
        f'<th>CV Score</th><th>±Std</th><th>Best Params</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _fi_html(fi_df: pd.DataFrame) -> str:
    if fi_df is None or fi_df.empty:
        return ""
    max_val = fi_df["Importance"].max() or 1
    rows = ""
    for _, row in fi_df.iterrows():
        pct = row["Importance"] / max_val * 100
        rows += (
            f'<tr>'
            f'<td><code class="feat">{row["Feature"]}</code></td>'
            f'<td class="num">{row["Importance"]:.4f}</td>'
            f'<td class="num muted">{pct:.1f}%</td>'
            f'<td style="min-width:180px;padding-right:20px">'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            f'</td>'
            f'</tr>'
        )
    return (
        f'<table class="tbl">'
        f'<thead><tr><th>Feature</th><th>Importance</th><th>%</th><th>Relative</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _pred_html(y_test, y_pred, task_type: str) -> str:
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    n = len(y_test)

    if task_type == "classification":
        correct   = int((y_test == y_pred).sum())
        incorrect = n - correct
        acc       = correct / n * 100
        summary = (
            f'<div class="stat-row">'
            f'<div class="stat"><span class="stat-val ok">{correct:,}</span><span class="stat-lbl">Correct</span></div>'
            f'<div class="stat"><span class="stat-val ng">{incorrect:,}</span><span class="stat-lbl">Incorrect</span></div>'
            f'<div class="stat"><span class="stat-val hi">{acc:.1f}%</span><span class="stat-lbl">Accuracy</span></div>'
            f'<div class="stat"><span class="stat-val muted">{n:,}</span><span class="stat-lbl">Total</span></div>'
            f'</div>'
        )
        header = "<tr><th>#</th><th>Actual</th><th>Predicted</th><th style='width:64px;text-align:center'>Result</th></tr>"
        body   = "".join(
            f'<tr>'
            f'<td class="row-idx">{i+1}</td>'
            f'<td>{a}</td><td>{p}</td>'
            f'<td style="text-align:center">{"<span class=\'ok bold\'>✓</span>" if a==p else "<span class=\'ng bold\'>✗</span>"}</td>'
            f'</tr>'
            for i, (a, p) in enumerate(zip(y_test[:50], y_pred[:50]))
        )
    else:
        errs = np.abs(y_pred - y_test)
        summary = (
            f'<div class="stat-row">'
            f'<div class="stat"><span class="stat-val hi">{errs.mean():.4f}</span><span class="stat-lbl">MAE</span></div>'
            f'<div class="stat"><span class="stat-val muted">{errs.max():.4f}</span><span class="stat-lbl">Max Error</span></div>'
            f'<div class="stat"><span class="stat-val muted">{errs.min():.4f}</span><span class="stat-lbl">Min Error</span></div>'
            f'<div class="stat"><span class="stat-val muted">{n:,}</span><span class="stat-lbl">Total</span></div>'
            f'</div>'
        )
        header = "<tr><th>#</th><th>Actual</th><th>Predicted</th><th>Error</th><th>Abs Error</th></tr>"
        body   = "".join(
            f'<tr>'
            f'<td class="row-idx">{i+1}</td>'
            f'<td class="num">{a:.4f}</td><td class="num">{p:.4f}</td>'
            f'<td class="num {"ng" if (p-a)>0 else "ok"}">{p-a:+.4f}</td>'
            f'<td class="num">{abs(p-a):.4f}</td>'
            f'</tr>'
            for i, (a, p) in enumerate(zip(y_test[:50], y_pred[:50]))
        )

    note = f'<p class="table-note">Showing first 50 of {n:,} rows</p>'
    return (
        f'{summary}'
        f'{note}'
        f'<table class="tbl"><thead>{header}</thead><tbody>{body}</tbody></table>'
    )


# ── Main builder ──────────────────────────────────────────────────────────────

def build_html_report(result: dict, metrics: dict, fi_df=None) -> str:
    now         = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_type   = result["task_type"]
    best_label  = result["best_label"]
    best_params = result.get("best_params") or {}
    n_models    = len(result["competition"])
    has_fi      = fi_df is not None and not fi_df.empty

    # metric cards
    met_cards = "".join(
        f'<div class="mc"><div class="mc-lbl">{k}</div><div class="mc-val">{v:.4f}</div></div>'
        for k, v in metrics.items()
    )

    # params table
    params_rows = "".join(
        f'<tr><td><code class="feat">{k}</code></td><td class="num">{v}</td></tr>'
        for k, v in best_params.items()
    )
    params_tbl = (
        f'<table class="tbl"><thead><tr><th>Parameter</th><th>Value</th></tr></thead>'
        f'<tbody>{params_rows}</tbody></table>'
        if best_params else '<p class="muted" style="padding:12px 0">Default parameters</p>'
    )

    fi_section_html = ""
    fi_nav          = ""
    pred_num        = "04"
    if has_fi:
        fi_section_html = f"""
      <section id="fi">
        <div class="sec-hd"><span class="sec-num">04</span><div><div class="sec-title">Feature Importance</div><div class="sec-sub">feature ที่มีผลต่อการตัดสินใจของ model มากที่สุด</div></div></div>
        {_fi_html(fi_df)}
      </section>"""
        fi_nav   = '<a href="#fi"><span class="nav-num">04</span>Feature Importance</a>'
        pred_num = "05"

    # summary strip values
    strip = "".join(
        f'<div class="ss"><div class="ss-val">{v:.4f}</div><div class="ss-lbl">{k}</div></div>'
        for k, v in metrics.items()
    )
    strip += (
        f'<div class="ss"><div class="ss-val">{n_models}</div><div class="ss-lbl">Models Tested</div></div>'
    )

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Report — {best_label}</title>
<style>
/* Reset */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}

/* Base */
body{{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Sarabun',sans-serif;
  background:#0d1117;color:#e6edf3;font-size:14px;line-height:1.6;
  display:flex;min-height:100vh;
}}

/* ── Sidebar ─────────────────────────── */
.sidebar{{
  width:200px;flex-shrink:0;position:sticky;top:0;height:100vh;
  background:#010409;border-right:1px solid #21262d;
  padding:0;overflow-y:auto;display:flex;flex-direction:column;
}}
.sidebar-brand{{
  padding:24px 20px 18px;border-bottom:1px solid #21262d;
  font-size:.68rem;font-weight:800;letter-spacing:.14em;
  text-transform:uppercase;color:#8b949e;
}}
.sidebar nav{{padding:12px 0;flex:1}}
.sidebar a{{
  display:flex;align-items:center;gap:10px;
  padding:8px 20px;color:#8b949e;text-decoration:none;
  font-size:.8rem;transition:color .15s,background .15s;
  border-left:2px solid transparent;
}}
.sidebar a:hover{{color:#e6edf3;background:#0d1117;border-left-color:#388bfd}}
.nav-num{{
  font-size:.65rem;font-weight:700;color:#388bfd44;
  background:#1f3a5f22;padding:1px 5px;border-radius:4px;
  flex-shrink:0;font-variant-numeric:tabular-nums;
}}
.sidebar-footer{{
  padding:16px 20px;border-top:1px solid #21262d;
  font-size:.72rem;color:#4b5563;line-height:1.5;
}}

/* ── Main ────────────────────────────── */
.main{{flex:1;padding:44px 56px;min-width:0}}

/* ── Page header ─────────────────────── */
.page-title{{font-size:1.5rem;font-weight:800;color:#e6edf3;margin-bottom:6px}}
.page-meta{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:32px}}
.badge{{
  display:inline-flex;align-items:center;
  padding:3px 10px;border-radius:20px;
  font-size:.72rem;font-weight:700;letter-spacing:.04em;
}}
.badge-task{{background:#1f3a5f;color:#58a6ff;border:1px solid #388bfd33}}
.badge-model{{background:#1a3a2a;color:#3fb950;border:1px solid #2ea04333}}
.meta-time{{color:#4b5563;font-size:.78rem;margin-left:4px}}

/* ── Summary strip ───────────────────── */
.strip{{
  display:flex;flex-wrap:wrap;gap:0;
  background:#161b22;border:1px solid #30363d;border-radius:10px;
  margin-bottom:44px;overflow:hidden;
}}
.ss{{
  flex:1;min-width:100px;padding:18px 20px;
  border-right:1px solid #21262d;text-align:center;
}}
.ss:last-child{{border-right:none}}
.ss-val{{font-size:1.25rem;font-weight:800;color:#e6edf3;font-variant-numeric:tabular-nums}}
.ss-lbl{{font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.07em;margin-top:3px}}

/* ── Sections ────────────────────────── */
section{{margin-bottom:52px}}
.sec-hd{{display:flex;align-items:flex-start;gap:12px;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid #21262d}}
.sec-num{{
  font-size:.65rem;font-weight:800;color:#58a6ff;
  background:#1f3a5f;border:1px solid #388bfd33;
  border-radius:5px;padding:3px 7px;flex-shrink:0;
  margin-top:1px;letter-spacing:.04em;
}}
.sec-title{{font-size:1rem;font-weight:700;color:#e6edf3}}
.sec-sub{{font-size:.78rem;color:#8b949e;margin-top:2px}}

/* ── Two-col grid ────────────────────── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.col-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px}}
.col-card-title{{font-size:.72rem;font-weight:700;text-transform:uppercase;
                 letter-spacing:.08em;color:#8b949e;margin-bottom:14px}}

/* ── Metric cards ────────────────────── */
.mc-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px}}
.mc{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 16px}}
.mc-lbl{{font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px}}
.mc-val{{font-size:1.2rem;font-weight:800;color:#58a6ff;font-variant-numeric:tabular-nums}}

/* ── Tables ──────────────────────────── */
.tbl{{width:100%;border-collapse:collapse;font-size:.835rem}}
.tbl th{{
  background:#0d1117;color:#8b949e;font-weight:600;
  text-align:left;padding:9px 12px;
  border-bottom:1px solid #30363d;
  font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
}}
.tbl td{{padding:9px 12px;border-bottom:1px solid #21262d;vertical-align:middle}}
.tbl tbody tr:hover td{{background:#161b2266}}
.num{{font-variant-numeric:tabular-nums;text-align:right}}
.muted{{color:#8b949e}}
.row-best td{{background:#111f16}}
.row-error{{opacity:.45}}
.row-idx{{color:#8b949e;font-size:.78rem;width:36px}}
.params-cell{{font-size:.75rem;color:#8b949e;max-width:280px}}
.error-msg{{color:#f85149;font-size:.78rem}}

/* ── Rank / badges ───────────────────── */
.rank-top{{
  display:inline-block;width:22px;height:22px;line-height:22px;
  text-align:center;border-radius:50%;font-size:.72rem;font-weight:800;
  background:#1f3a5f;color:#58a6ff;
}}
.rank-num{{color:#8b949e;font-size:.8rem;font-weight:600}}
.model-name{{font-weight:600}}
.best-tag{{
  font-size:.65rem;font-weight:700;letter-spacing:.05em;
  background:#1a3a2a;color:#3fb950;border:1px solid #2ea04333;
  padding:1px 7px;border-radius:10px;margin-left:8px;
}}

/* ── Feature bar ─────────────────────── */
.bar-bg{{background:#21262d;border-radius:3px;height:6px}}
.bar-fill{{background:#388bfd;height:6px;border-radius:3px;min-width:2px}}
.feat{{color:#79c0ff;font-family:ui-monospace,'SFMono-Regular',monospace;font-size:.8rem}}

/* ── Prediction summary ──────────────── */
.stat-row{{
  display:flex;gap:0;background:#161b22;border:1px solid #30363d;
  border-radius:10px;margin-bottom:16px;overflow:hidden;
}}
.stat{{flex:1;padding:16px 20px;text-align:center;border-right:1px solid #21262d}}
.stat:last-child{{border-right:none}}
.stat-val{{display:block;font-size:1.3rem;font-weight:800;font-variant-numeric:tabular-nums}}
.stat-lbl{{display:block;font-size:.68rem;color:#8b949e;text-transform:uppercase;letter-spacing:.07em;margin-top:3px}}
.ok{{color:#3fb950}}.ng{{color:#f85149}}.hi{{color:#58a6ff}}.bold{{font-weight:700}}
.table-note{{color:#8b949e;font-size:.75rem;margin-bottom:8px}}
</style>
</head>
<body>

<aside class="sidebar">
  <div class="sidebar-brand">ML Report</div>
  <nav>
    <a href="#overview"><span class="nav-num">—</span>Overview</a>
    <a href="#perf"><span class="nav-num">01</span>Performance</a>
    <a href="#leaderboard"><span class="nav-num">02</span>Leaderboard</a>
    {fi_nav}
    <a href="#predictions"><span class="nav-num">{pred_num}</span>Predictions</a>
  </nav>
  <div class="sidebar-footer">{best_label}<br>{task_type.title()}<br>{now}</div>
</aside>

<main class="main">

  <div class="page-title">ML Pipeline Report</div>
  <div class="page-meta">
    <span class="badge badge-task">{task_type.title()}</span>
    <span class="badge badge-model">{best_label}</span>
    <span class="meta-time">{now}</span>
  </div>

  <!-- Overview strip -->
  <div id="overview" class="strip">{strip}</div>

  <!-- 01 · Performance -->
  <section id="perf">
    <div class="sec-hd">
      <span class="sec-num">01</span>
      <div>
        <div class="sec-title">Model Performance</div>
        <div class="sec-sub">ผลการประเมินบน test set ที่ model ไม่เคยเห็น</div>
      </div>
    </div>
    <div class="two-col">
      <div class="col-card">
        <div class="col-card-title">Metrics</div>
        <div class="mc-grid">{met_cards}</div>
      </div>
      <div class="col-card">
        <div class="col-card-title">Hyperparameters — {best_label}</div>
        {params_tbl}
      </div>
    </div>
  </section>

  <!-- 02 · Leaderboard -->
  <section id="leaderboard">
    <div class="sec-hd">
      <span class="sec-num">02</span>
      <div>
        <div class="sec-title">Model Leaderboard</div>
        <div class="sec-sub">ผลการแข่งขัน {n_models} model เรียงตาม CV Score</div>
      </div>
    </div>
    {_leaderboard_html(result["competition"], best_label)}
  </section>

  {fi_section_html}

  <!-- {pred_num} · Predictions -->
  <section id="predictions">
    <div class="sec-hd">
      <span class="sec-num">{pred_num}</span>
      <div>
        <div class="sec-title">Predictions</div>
        <div class="sec-sub">ตัวอย่างผลการทำนายเปรียบเทียบกับค่าจริง</div>
      </div>
    </div>
    {_pred_html(result["y_test"], result["y_pred"], task_type)}
  </section>

</main>
</body>
</html>"""
