# --- Spendo! v3.1.1 ---
from __future__ import annotations
import hashlib, io, re, zipfile, random, colorsys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from dateutil import parser as du
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "budget.db"
RULES_DIR = APP_DIR / "rules"
HELP_DIR = APP_DIR / "help"
BACKUPS_DIR = APP_DIR / "backups"

CATS_FILE = RULES_DIR / "categories.yaml"
RECUR_FILE = RULES_DIR / "recurring.yaml"
BUDGETS_FILE = RULES_DIR / "budgets.yaml"
ALIASES_FILE = RULES_DIR / "aliases.yaml"
PRESETS_FILE = RULES_DIR / "bank_presets.yaml"
SETTINGS_FILE = RULES_DIR / "settings.yaml"

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

# ---------- IO helpers ----------
def load_yaml(path: Path, default):
    try:
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or default
    except Exception:
        pass
    return default

def save_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

def ensure_tables():
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                transaction_id TEXT,
                date TIMESTAMP,
                description TEXT,
                amount REAL,
                currency TEXT,
                category TEXT,
                hash_key TEXT UNIQUE,
                raw_date_text TEXT
            );
            """
        )
        cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info(transactions)").fetchall()]
        if "raw_date_text" not in cols:
            conn.exec_driver_sql("ALTER TABLE transactions ADD COLUMN raw_date_text TEXT")

# ---------- Notifications ----------
def add_toast(kind: str, msg: str, key: Optional[str] = None, persist: int = 4):
    key = key or f"toast_{hashlib.sha1(msg.encode()).hexdigest()[:8]}"
    ts = st.session_state.setdefault("_toasts", {})
    ts[key] = {"kind": kind, "msg": msg, "left": int(max(1, persist))}

def render_toasts():
    ts = st.session_state.get("_toasts", {})
    drop = []
    for k, v in ts.items():
        if v["kind"] == "success": st.success(v["msg"])
        elif v["kind"] == "info": st.info(v["msg"])
        elif v["kind"] == "warning": st.warning(v["msg"])
        else: st.error(v["msg"])
        v["left"] -= 1
        if v["left"] <= 0: drop.append(k)
    for k in drop: ts.pop(k, None)

def flash_here(container, kind: str, msg: str):
    if kind == "success": container.success(msg)
    elif kind == "info": container.info(msg)
    elif kind == "warning": container.warning(msg)
    else: container.error(msg)

# ---------- Color helpers (HEX only) ----------
HEX_RE = re.compile(r"^#([0-9a-fA-F]{6})$")

def random_pastel_hex() -> str:
    import colorsys, random
    h = random.random(); s = 0.45; v = 0.85
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def to_hex_color(c: Optional[str]) -> str:
    if isinstance(c, str) and HEX_RE.match(c): return c
    return random_pastel_hex()

def sanitize_category_colors(colors: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for k, v in (colors or {}).items():
        out[str(k)] = to_hex_color(v)
    return out

# ---------- Parsing / normalizing ----------
def normalize_amount(x):
    if pd.isna(x): return None
    s = str(x).strip().replace("CHF","").replace("chf","").replace(" ","").replace("'","").replace(",",".").replace("+","")
    try: return float(s)
    except Exception: return None

def parse_date(s, *, dayfirst=True, yearfirst=False, fmt=None):
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    s = str(s).strip()
    if not s: return None
    if fmt:
        try: return datetime.strptime(s, fmt).date()
        except Exception: return None
    try: return du.parse(s, dayfirst=dayfirst, yearfirst=yearfirst).date()
    except Exception: return None

def month_bounds(d: date):
    first = d.replace(day=1)
    last = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return first, last

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def resolve_col(df: pd.DataFrame, user_key: str, synonyms: List[str], required: bool = True) -> Optional[str]:
    if df is None or df.empty: return None
    norm_map = {re.sub(r"\s+", " ", c).strip().lower(): c for c in df.columns}
    def find(name: str) -> Optional[str]:
        k = re.sub(r"\s+", " ", str(name)).strip().lower()
        return norm_map.get(k)
    if user_key:
        got = find(user_key)
        if got: return got
    for s in synonyms:
        got = find(s)
        if got: return got
    if required:
        add_toast("error", f"Missing required column. Tried: “{user_key}” or {synonyms}. Available: {list(df.columns)}")
    return None

# ---------- Auto-detect preset ----------
def select_best_preset(filename: str, df_in: pd.DataFrame, presets: dict) -> Optional[dict]:
    if not presets or "presets" not in presets: return None
    fname = (filename or "").lower()
    headers = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df_in.columns]
    best = (None, -1)
    for p in presets["presets"]:
        s = 0
        m = (p.get("match") or {})
        for kw in (m.get("filename_contains") or []):
            if str(kw).lower() in fname: s += 2
        for hh in (m.get("header_contains") or []):
            if str(hh).strip().lower() in headers: s += 1
        if s > best[1]: best = (p, s)
    return best[0] if best[1] >= 2 else None

# ---------- Editors helpers ----------
def list_to_csv(lst: List[str] | None) -> str:
    return ", ".join([s for s in (lst or []) if str(s).strip()])

def csv_to_list(s: str) -> List[str]:
    return [t.strip() for t in str(s).split(",") if t.strip()]

# ---------- Categorisation ----------
def apply_aliases(text: str, aliases) -> str:
    if not text: return text
    t = str(text)
    for a in aliases.get("aliases", []):
        find, repl = a.get("find"), a.get("replace")
        if find and repl and find.lower() in t.lower():
            t = re.sub(re.escape(find), repl, t, flags=re.IGNORECASE)
    return t

def apply_categories(df: pd.DataFrame, cats: dict, aliases: dict) -> pd.DataFrame:
    if df.empty: return df
    if "description" in df.columns:
        df["description"] = df["description"].apply(lambda s: apply_aliases(s, aliases))

    compiled = []
    for idx, r in enumerate(cats.get("rules", [])):
        cat = r.get("category", "Other")
        pr = int(r.get("priority", 0))
        kws = [str(k) for k in r.get("keywords", []) if str(k).strip()]
        negs = [str(k) for k in r.get("negative_keywords", []) if str(k).strip()]
        use_regex = bool(r.get("regex", False))
        pos_patterns = [re.compile(kw if use_regex else re.escape(kw), re.IGNORECASE) for kw in kws]
        neg_patterns = [re.compile(ng if use_regex else re.escape(ng), re.IGNORECASE) for ng in negs]
        compiled.append({
            "idx": idx, "category": cat, "priority": pr,
            "keywords": kws, "pos": pos_patterns, "neg": neg_patterns,
        })

    out_cat, why_kw, why_rule_idx, why_priority = [], [], [], []
    for _, row in df.iterrows():
        text = re.sub(r"\s+"," ", str(row.get("description","")).casefold()).strip()
        best = dict(cat="Other", pr=-1, kw_len=-1, kw="", idx=-1)
        for r in compiled:
            if any(p.search(text) for p in r["neg"]): continue
            for pat, kw in zip(r["pos"], r["keywords"]):
                if pat.search(text):
                    score = (r["priority"], len(kw))
                    if score > (best["pr"], best["kw_len"]):
                        best = dict(cat=r["category"], pr=r["priority"], kw_len=len(kw), kw=kw, idx=r["idx"])
                    break
        out_cat.append(best["cat"]); why_kw.append(best["kw"] or "(no match)")
        why_rule_idx.append(best["idx"]); why_priority.append(best["pr"])
    df["category"] = out_cat
    df["_match_reason"] = why_kw
    df["_match_rule_idx"] = why_rule_idx
    df["_match_priority"] = why_priority
    return df

# ---------- Re-categorize entire DB ----------
def recategorize_all_transactions(cats: dict, aliases: dict, settings: dict) -> int:
    with engine.begin() as conn:
        df_all = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
    if df_all.empty: return 0
    before = df_all["category"].astype(str).tolist()
    df_new = apply_categories(df_all.copy(), cats, aliases)
    if settings.get("auto_sign_detection", True):
        cat_signs = settings.get("category_signs", {})
        def fix_sign(row):
            cat = row.get("category", "Other")
            sign = cat_signs.get(cat, -1)
            amt = abs(float(row["amount"])) if pd.notna(row["amount"]) else row["amount"]
            return amt if sign >= 0 else -abs(amt)
        df_new["amount"] = df_new.apply(fix_sign, axis=1)
    changed = 0
    with engine.begin() as conn:
        for i, r in df_new.iterrows():
            if r.get("category") != before[i]:
                changed += 1
            conn.exec_driver_sql(
                "UPDATE transactions SET category=?, amount=? WHERE id=?",
                (r.get("category"), float(r.get("amount")), int(r.get("id")))
            )
    return changed

# ---------- Backups ----------
def auto_backup():
    try:
        BACKUPS_DIR.mkdir(exist_ok=True, parents=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zpath = BACKUPS_DIR / f"backup_{ts}.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            if DB_PATH.exists(): z.write(DB_PATH, arcname="budget.db")
            for y in ["categories.yaml","recurring.yaml","budgets.yaml","aliases.yaml","bank_presets.yaml","settings.yaml"]:
                p = RULES_DIR / y
                if p.exists(): z.write(p, arcname=f"rules/{y}")
        return zpath.name
    except Exception:
        return None

# ---------- App setup ----------
st.set_page_config(page_title="Spendo!", layout="wide")
ensure_tables()

# State flags
if "refresh_transactions" not in st.session_state:
    st.session_state["refresh_transactions"] = False

# Load config
cats = load_yaml(CATS_FILE, {"rules": []})
recur = load_yaml(RECUR_FILE, {"bills": []})
budgets = load_yaml(BUDGETS_FILE, {"budgets": []})
aliases = load_yaml(ALIASES_FILE, {"aliases": []})
settings = load_yaml(SETTINGS_FILE, {
    "auto_sign_detection": True,
    "category_signs": {"Income": 1, "Other": -1},
    "category_colors": {},
    # theme key is ignored now; app is dark-only
    "date_parse": {"dayfirst": True, "yearfirst": False},
})
settings["category_colors"] = sanitize_category_colors(settings.get("category_colors", {}))
save_yaml(SETTINGS_FILE, settings)

# Force dark theme always
plotly_theme = "plotly_dark"
st.markdown("<style>.stApp{background:#0e1117;color:#e6e6e6}</style>", unsafe_allow_html=True)

# Header (no theme toggle)
st.title("Spendo v3.1.1")
render_toasts()

# ---------- TABS ----------
tab_dash, tab_tx, tab_rec, tab_upload, tab_settings, tab_help = st.tabs(
    ["Dashboard", "Transactions", "Upcoming & Recurring", "Upload & Map", "Settings", "Help"]
)

# ================== Dashboard ==================
with tab_dash:
    st.header("Dashboard")
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])

    if df.empty:
        st.info("No data yet. Go to **Upload & Map** to import a CSV.")
    else:
        today = date.today()
        kpi_first, kpi_last = month_bounds(today)
        kpi_df = df[(df["date"] >= pd.Timestamp(kpi_first)) & (df["date"] <= pd.Timestamp(kpi_last))].copy()

        view = st.radio("View", ["Net", "Expenses", "Income"], horizontal=True, key="dash_view")
        if view == "Expenses":
            kpi_amt = kpi_df[kpi_df["amount"] < 0]["amount"].sum()
        elif view == "Income":
            kpi_amt = kpi_df[kpi_df["amount"] > 0]["amount"].sum()
        else:
            kpi_amt = kpi_df["amount"].sum()

        c1,c2,c3 = st.columns(3)
        c1.metric("This month total", f"{kpi_amt:,.2f} CHF")
        avg6 = (
            df[df["date"] < pd.Timestamp(kpi_first)]
            .assign(m=df["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("m", as_index=False)["amount"].sum()
            .sort_values("m").tail(6)["amount"].mean()
            if not df.empty else 0.0
        )
        c2.metric("6-mo average net", f"{(avg6 or 0.0):,.2f} CHF")
        c3.metric("Transactions (this month)", f"{len(kpi_df):,}")

        # 12M single-series bar depending on View
        roll = (
            df.assign(m=df["date"].dt.to_period("M").dt.to_timestamp())
              .groupby("m", as_index=False)
              .agg(
                  net=("amount","sum"),
                  expenses=("amount", lambda s: s[s<0].abs().sum()),
                  income=("amount", lambda s: s[s>0].sum())
              )
              .sort_values("m").tail(12)
        )
        if not roll.empty:
            if view == "Expenses":
                series_name = "Expenses (abs)"
                plot_df = roll[["m","expenses"]].rename(columns={"expenses":"amount"})
            elif view == "Income":
                series_name = "Income"
                plot_df = roll[["m","income"]].rename(columns={"income":"amount"})
            else:
                series_name = "Net"
                plot_df = roll[["m","net"]].rename(columns={"net":"amount"})
            fig_trend = px.bar(plot_df, x="m", y="amount", title=f"{series_name} — last 12 months")
            fig_trend.update_layout(template=plotly_theme, xaxis_title="", yaxis_title="Amount", showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)

        # Donut: independent month picker
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        months = sorted(df["month"].dropna().unique().tolist(), reverse=True)
        donut_ref = st.selectbox("Donut month", months, index=0, format_func=lambda x: x.strftime("%b %Y"))
        d_first, d_last = month_bounds(donut_ref.date())
        donut_df = df[(df["date"] >= pd.Timestamp(d_first)) & (df["date"] <= pd.Timestamp(d_last))].copy()
        exp = donut_df[donut_df["amount"] < 0].copy()
        if not exp.empty:
            cat = exp.groupby(["category"], as_index=False)["amount"].sum()
            cat["abs_amount"] = cat["amount"].abs()
            cat["color"] = cat["category"].apply(lambda c: settings.get("category_colors", {}).get(c, "#95a5a6"))
            fig = px.pie(
                cat, values="abs_amount", names="category", hole=0.45,
                title=f"Spending by category ({donut_ref.strftime('%b %Y')})",
                color="category", color_discrete_sequence=cat["color"],
            )
            fig.update_layout(template=plotly_theme)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expenses in the selected month.")

# ================== Transactions ==================
with tab_tx:
    if st.session_state.get("refresh_transactions"):
        st.session_state["refresh_transactions"] = False
        st.experimental_rerun()

    st.header("Transactions")
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM transactions ORDER BY date DESC", conn, parse_dates=["date"])

    if df.empty:
        st.info("No data yet.")
    else:
        filters = st.columns([1,1,1,1,1])
        with filters[0]: start_d = st.date_input("Start date", value=df["date"].min().date(), key="tx_start")
        with filters[1]: end_d   = st.date_input("End date", value=df["date"].max().date(), key="tx_end")
        with filters[2]: catf    = st.selectbox("Category", ["(all)"]+sorted(df["category"].dropna().unique().tolist()), key="tx_cat")
        with filters[3]:
            currencies = sorted(df.get("currency", pd.Series(dtype=str)).dropna().unique().tolist())
            curf = st.selectbox("Currency", ["(all)"]+currencies, key="tx_cur")
        with filters[4]: search  = st.text_input("Search", key="tx_search")

        modes = st.columns([1,1])
        with modes[0]:
            edit_mode = st.toggle("Edit mode (inline)", value=False, key="tx_edit")
        with modes[1]:
            delete_mode = st.toggle("Delete mode", value=False, key="tx_delete")

        mask = (df["date"].dt.date>=start_d) & (df["date"].dt.date<=end_d)
        if catf!="(all)": mask &= (df["category"]==catf)
        if curf!="(all)" and "currency" in df.columns: mask &= (df["currency"]==curf)
        if search: mask &= df["description"].str.contains(search, case=False, na=False)
        fdf = df[mask].copy()

        fdf["date_str"] = fdf["date"].dt.strftime("%d.%m.%Y")
        base_cols = ["id","source_file","date","description","amount","currency","category"]
        work = fdf[base_cols].copy()
        work["date"] = pd.to_datetime(work["date"]).dt.date

        if not delete_mode and not edit_mode:
            show_cols = ["id","date_str","description","amount","currency","category","source_file"]
            def style_rows(row):
                bg = settings.get("category_colors", {}).get(row.get("category","Other"), "#95a5a6")
                return [f"background-color:{bg}; color:white;" for _ in row]
            st.dataframe(
                fdf[show_cols].style.apply(style_rows, axis=1),
                use_container_width=True, hide_index=True
            )
        else:
            if delete_mode:
                work["Delete"] = False
            editor_cols = ["id","source_file","date","description","amount","currency","category"] + (["Delete"] if delete_mode else [])
            editable_cols = ["date","description","amount","currency","category"]
            allowed_edit = set()
            if edit_mode: allowed_edit.update(editable_cols)
            if delete_mode: allowed_edit.add("Delete")
            disabled_cols = [c for c in editor_cols if c not in allowed_edit]
            colcfg = {
                "date": st.column_config.DateColumn("date"),
                "amount": st.column_config.NumberColumn("amount", step=0.01),
                "description": st.column_config.TextColumn("description"),
                "currency": st.column_config.TextColumn("currency"),
                "category": st.column_config.TextColumn("category"),
                "id": st.column_config.TextColumn("id", disabled=True),
                "source_file": st.column_config.TextColumn("source_file", disabled=True),
            }
            if delete_mode:
                colcfg["Delete"] = st.column_config.CheckboxColumn("Delete")

            edited = st.data_editor(
                work[editor_cols],
                use_container_width=True, hide_index=True, key="tx_editor_main",
                disabled=disabled_cols, column_config=colcfg
            )

            edit_note = st.empty()
            del_note = st.empty()

            if edit_mode and st.button("Save edits", key="save_tx_edits"):
                merged = edited.merge(
                    work[["id","date","description","amount","currency","category"]].rename(columns={
                        "date":"date_old","description":"description_old","amount":"amount_old",
                        "currency":"currency_old","category":"category_old"
                    }),
                    on="id", how="left"
                )
                updates = []
                for _, r in merged.iterrows():
                    changes = {}
                    if pd.notna(r.get("date")) and r["date"] != r["date_old"]:
                        changes["date"] = pd.Timestamp(r["date"])
                    if str(r.get("description")) != str(r["description_old"]):
                        changes["description"] = str(r.get("description"))
                    new_amt = normalize_amount(r.get("amount"))
                    old_amt = normalize_amount(r.get("amount_old"))
                    if (new_amt is not None) and (new_amt != old_amt):
                        changes["amount"] = float(new_amt)
                    if str(r.get("currency")) != str(r["currency_old"]):
                        changes["currency"] = str(r.get("currency"))
                    if str(r.get("category")) != str(r["category_old"]):
                        changes["category"] = str(r.get("category"))
                    if changes:
                        updates.append((int(r["id"]), changes))
                if updates:
                    with engine.begin() as conn:
                        for rid, ch in updates:
                            sets = ", ".join([f"{k}=?" for k in ch.keys()])
                            vals = list(ch.values()) + [rid]
                            conn.exec_driver_sql(f"UPDATE transactions SET {sets} WHERE id=?", tuple(vals))
                    flash_here(edit_note, "success", f"Updated {len(updates)} row(s).")
                    add_toast("success", f"Updated {len(updates)} row(s).", key="tx_updates")
                    st.session_state["refresh_transactions"] = True
                else:
                    flash_here(edit_note, "info", "No changes to save.")
                    add_toast("info", "No changes to save.", key="tx_no_changes")

            if delete_mode:
                to_delete = edited[edited.get("Delete", False) == True]["id"].tolist() if "Delete" in edited.columns else []
                if st.button(f"Delete selected rows ({len(to_delete)})", type="primary", disabled=(len(to_delete)==0), key="btn_del_rows_main"):
                    with engine.begin() as conn:
                        if to_delete:
                            conn.exec_driver_sql(
                                f"DELETE FROM transactions WHERE id IN ({','.join(['?']*len(to_delete))})",
                                tuple(to_delete)
                            )
                    flash_here(del_note, "success", f"Deleted {len(to_delete)} row(s).")
                    add_toast("success", f"Deleted {len(to_delete)} row(s).", key="tx_deleted")
                    st.session_state["refresh_transactions"] = True

        # Category colors editor
        with st.expander("Category colors", expanded=False):
            all_cats = sorted(set(df["category"].dropna().unique().tolist()) | set(settings.get("category_colors", {}).keys()))
            changed = False
            for cc in all_cats:
                default = to_hex_color(settings.get("category_colors", {}).get(cc, "#95a5a6"))
                newc = st.color_picker(cc, default, key=f"col_{cc}")
                if newc != default:
                    settings.setdefault("category_colors", {})[cc]=to_hex_color(newc); changed=True
            if changed:
                save_yaml(SETTINGS_FILE, settings); add_toast("success", "Saved category colors.", key="saved_cat_colors")

        # --- Why inspector (Summary + Nerd mode; no sample rule) ---
        with st.expander("Why was this categorized this way?", expanded=False):
            if not fdf.empty:
                fdf_reason = apply_categories(fdf.copy(), cats, aliases)
                col_why1, col_why2 = st.columns([1, 2])
                with col_why1:
                    tx_id = st.selectbox("Pick a transaction", fdf_reason["id"].tolist(), key="why_tx_id")
                with col_why2:
                    tx = fdf_reason.loc[fdf_reason["id"] == tx_id].iloc[0]
                    st.write(f"**Description:** {tx['description']}")
                    st.write(f"**Category (current DB):** {fdf.loc[fdf['id']==tx_id, 'category'].iloc[0]}")
                    st.write(f"**Keyword matched:** `{tx.get('_match_reason', '(n/a)')}`")

                with st.expander("Nerd mode", expanded=False):
                    rule_idx = int(tx.get("_match_rule_idx", -1))
                    if 0 <= rule_idx < len(cats.get("rules", [])):
                        r = cats["rules"][rule_idx]
                        st.code(yaml.safe_dump(r, allow_unicode=True, sort_keys=False), language="yaml")
                    else:
                        st.caption("No rule matched by the current rules; categorization defaulted.")
            else:
                st.caption("No transactions in the current filter.")

        st.download_button(
            "Export filtered CSV",
            fdf[["id","date","description","amount","currency","category","source_file"]].to_csv(index=False).encode("utf-8"),
            file_name="transactions_filtered.csv"
        )

# ================== Upcoming & Recurring ==================
with tab_rec:
    st.header("Upcoming & Recurring")
    bills = recur.get("bills", [])
    today = date.today()
    if not bills:
        st.info("No recurring bills configured. Add entries in **Settings → Rules → Recurring (bills)**.")
    else:
        rows=[]
        for b in bills:
            d = int(b.get("day",1))
            last_day = (today.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
            due = date(today.year, today.month, min(d,last_day.day))
            if due < today:
                y,m = (today.year+1,1) if today.month==12 else (today.year, today.month+1)
                last_next = (date(y,m,28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
                due = date(y,m, min(d,last_next.day))
            rows.append({"name":b.get("name"), "day":d, "amount":b.get("amount"), "due":due})
        st.dataframe(pd.DataFrame(rows).sort_values("due"), use_container_width=True)

# ================== Upload & Map ==================
with tab_upload:
    st.header("Upload & Map")

    presets_data = load_yaml(PRESETS_FILE, {"presets": []})
    preset_names = ["(auto-detect)"] + [p["name"] for p in presets_data.get("presets", [])]

    if "mapping_fields" not in st.session_state:
        st.session_state["mapping_fields"] = {"date":"date","amount":"amount","description":"description","currency":"currency","transaction_id":"transaction_id"}
    if "selected_preset" not in st.session_state:
        st.session_state["selected_preset"] = "(auto-detect)"

    topm = st.columns([2,1])
    with topm[0]:
        preset_name = st.selectbox(
            "Preset", preset_names,
            index=preset_names.index(st.session_state["selected_preset"]) if st.session_state["selected_preset"] in preset_names else 0
        )

    if preset_name != st.session_state["selected_preset"]:
        st.session_state["selected_preset"] = preset_name
        if preset_name != "(auto-detect)":
            chosen = next((p for p in presets_data.get("presets", []) if p["name"]==preset_name), None)
            if chosen and "mapping" in chosen:
                st.session_state["mapping_fields"].update(chosen["mapping"])

    m1,m2,m3 = st.columns(3)
    with m1:
        date_col = st.text_input("Date column", key="map_date", value=st.session_state["mapping_fields"].get("date","date"))
        amount_col = st.text_input("Amount column", key="map_amount", value=st.session_state["mapping_fields"].get("amount","amount"))
    with m2:
        desc_col = st.text_input("Description column", key="map_description", value=st.session_state["mapping_fields"].get("description","description"))
        currency_col = st.text_input("Currency column (optional)", key="map_currency", value=st.session_state["mapping_fields"].get("currency","currency"))
    with m3:
        id_col = st.text_input("Transaction ID (optional)", key="map_transaction_id", value=st.session_state["mapping_fields"].get("transaction_id","transaction_id"))
        st.caption("Selecting a preset auto-fills these fields.")

    delimiter_input = st.text_input("Delimiter (leave blank to auto/preset)", value="")

    def try_read_bytes(raw: bytes, delim: Optional[str]) -> Optional[pd.DataFrame]:
        try:
            if delim:
                df = pd.read_csv(io.BytesIO(raw), delimiter=delim)
            else:
                df = pd.read_csv(io.BytesIO(raw))
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None

    st.divider()
    uploads = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)
    PREVIEW_LIMIT = 50

    note_preview = st.empty()
    if st.button("Preview mapping", key="preview_mapping") and uploads:
        up = uploads[0]
        raw = up.read()

        df_in = None
        if delimiter_input.strip():
            df_in = try_read_bytes(raw, delimiter_input.strip())
        if df_in is None:
            for d in [",",";","|","\t"]:
                df_in = try_read_bytes(raw, d)
                if df_in is not None: break
        if df_in is None:
            flash_here(note_preview, "error", f"Failed to read {up.name} with common delimiters.")
            add_toast("error", f"Failed to read {up.name} with common delimiters.")
        else:
            df_in = normalize_headers(df_in)
            if st.session_state["selected_preset"] != "(auto-detect)":
                chosen = next((p for p in presets_data.get("presets", []) if p["name"]==st.session_state["selected_preset"]), None)
            else:
                chosen = select_best_preset(up.name, df_in, presets_data)

            prefs = settings.get("date_parse", {"dayfirst": True, "yearfirst": False})
            p_dayfirst = (chosen.get("dayfirst") if chosen and "dayfirst" in chosen else prefs.get("dayfirst", True))
            p_yearfirst = (chosen.get("yearfirst") if chosen and "yearfirst" in chosen else prefs.get("yearfirst", False))
            p_fmt = chosen.get("date_format") if chosen else None

            mapping = (chosen.get("mapping") if chosen else
                       {"date":date_col, "description":desc_col, "amount":amount_col, "currency":currency_col, "transaction_id":id_col})

            date_res = resolve_col(df_in, mapping.get("date"), ["Booking date","Transaction date","Date"])
            desc_res = resolve_col(df_in, mapping.get("description"), ["Description1","Details","Merchant","Description"])
            amt_res  = resolve_col(df_in, mapping.get("amount"), ["Individual amount","Amount","Debit","Credit","Value"])
            cur_res  = resolve_col(df_in, mapping.get("currency"), ["Currency","Curr"], required=False)
            id_res   = resolve_col(df_in, mapping.get("transaction_id"), ["Transaction no.","Reference","TransactionID"], required=False)

            if date_res and desc_res and amt_res:
                std = pd.DataFrame()
                std["raw_date_text"] = df_in[date_res].astype(str)
                std["date"] = std["raw_date_text"].apply(lambda v: parse_date(v, dayfirst=p_dayfirst, yearfirst=p_yearfirst, fmt=p_fmt))

                build_list = chosen.get("build_description") if chosen else None
                if build_list:
                    parts = []
                    for col in build_list:
                        col_res = resolve_col(df_in, col, [col], required=False)
                        if col_res: parts.append(df_in[col_res].astype(str))
                    std["description"] = pd.Series([" - ".join(x) for x in zip(*parts)]) if parts else df_in[desc_res].astype(str)
                else:
                    std["description"] = df_in[desc_res].astype(str)

                std["amount"] = df_in[amt_res].apply(normalize_amount)
                std["currency"] = df_in[cur_res].astype(str) if cur_res else "CHF"
                if id_res: std["transaction_id"] = df_in[id_res].astype(str)
                std = std.dropna(subset=["date","description","amount"]).copy()
                std = apply_categories(std, cats, aliases)

                preview = std.head(PREVIEW_LIMIT).copy()
                cols = list(preview.columns)
                def style_row(row):
                    color = to_hex_color(settings.get("category_colors", {}).get(row.get("category","Other"), "#95a5a6"))
                    return [f"background-color: {color}; color: white;" for _ in cols]
                st.caption(f"Showing first {min(PREVIEW_LIMIT, len(std))} rows (of {len(std)})")
                st.dataframe(preview.style.apply(style_row, axis=1), use_container_width=True)
                flash_here(note_preview, "success", f"Preset: {(chosen.get('name') if chosen else 'None')} — mapping preview ready.")
            else:
                flash_here(note_preview, "error", "Missing required columns for preview.")
                add_toast("error", "Missing required columns for preview.", key="preview_missing_cols")

    note_import = st.empty()
    if st.button("Import to database", key="import_btn") and uploads:
        good = bad = inserted = 0
        for up in uploads:
            raw = up.read()

            df_in = None
            if delimiter_input.strip():
                df_in = try_read_bytes(raw, delimiter_input.strip())
            if df_in is None:
                for d in [",",";","|","\t"]:
                    df_in = try_read_bytes(raw, d)
                    if df_in is not None: break
            if df_in is None:
                bad += 1
                add_toast("error", f"Failed to read {up.name} with common delimiters.", key=f"import_read_fail_{up.name}")
                continue

            df_in = normalize_headers(df_in)
            if st.session_state["selected_preset"] != "(auto-detect)":
                chosen = next((p for p in presets_data.get("presets", []) if p["name"]==st.session_state["selected_preset"]), None)
            else:
                chosen = select_best_preset(up.name, df_in, presets_data)

            prefs = settings.get("date_parse", {"dayfirst": True, "yearfirst": False})
            p_dayfirst = (chosen.get("dayfirst") if chosen and "dayfirst" in chosen else prefs.get("dayfirst", True))
            p_yearfirst = (chosen.get("yearfirst") if chosen and "yearfirst" in chosen else prefs.get("yearfirst", False))
            p_fmt = chosen.get("date_format") if chosen else None

            mapping = (chosen.get("mapping") if chosen else
                       {"date":st.session_state["mapping_fields"]["date"],
                        "description":st.session_state["mapping_fields"]["description"],
                        "amount":st.session_state["mapping_fields"]["amount"],
                        "currency":st.session_state["mapping_fields"]["currency"],
                        "transaction_id":st.session_state["mapping_fields"]["transaction_id"]})

            date_res = resolve_col(df_in, mapping.get("date"), ["Booking date","Transaction date","Date"])
            desc_res = resolve_col(df_in, mapping.get("description"), ["Description1","Details","Merchant","Description"])
            amt_res  = resolve_col(df_in, mapping.get("amount"), ["Individual amount","Amount","Debit","Credit","Value"])
            cur_res  = resolve_col(df_in, mapping.get("currency"), ["Currency","Curr"], required=False)
            id_res   = resolve_col(df_in, mapping.get("transaction_id"), ["Transaction no.","Reference","TransactionID"], required=False)
            if not (date_res and desc_res and amt_res):
                bad += 1; add_toast("error", f"Missing required columns in {up.name}.", key=f"import_missing_{up.name}"); continue

            std = pd.DataFrame()
            raw_date_series = df_in[date_res].astype(str)
            std["raw_date_text"] = raw_date_series
            std["date"] = raw_date_series.apply(lambda v: parse_date(v, dayfirst=p_dayfirst, yearfirst=p_yearfirst, fmt=p_fmt))

            build_list = chosen.get("build_description") if chosen else None
            if build_list:
                parts = []
                for col in build_list:
                    col_res = resolve_col(df_in, col, [col], required=False)
                    if col_res: parts.append(df_in[col_res].astype(str))
                std["description"] = pd.Series([" - ".join(x) for x in zip(*parts)]) if parts else df_in[desc_res].astype(str)
            else:
                std["description"] = df_in[desc_res].astype(str)

            std["amount"] = df_in[amt_res].apply(normalize_amount)
            std["currency"] = df_in[cur_res].astype(str) if cur_res else "CHF"
            if id_res: std["transaction_id"] = df_in[id_res].astype(str)
            std = std.dropna(subset=["date","description","amount"]).copy()

            def row_hash(r):
                key = f"{up.name}|{r['date']}|{r['description']}|{r['amount']}|{r.get('currency','')}"
                return hashlib.sha256(key.encode("utf-8")).hexdigest()
            std["source_file"] = up.name
            std["hash_key"] = std.apply(row_hash, axis=1)

            std = apply_categories(std, cats, aliases)
            if settings.get("auto_sign_detection", True):
                cat_signs = settings.get("category_signs", {})
                def fix_sign(row):
                    cat = row.get("category","Other"); sign = cat_signs.get(cat, -1)
                    amt = abs(float(row["amount"])) if pd.notna(row["amount"]) else row["amount"]
                    return amt if sign >= 0 else -abs(amt)
                std["amount"] = std.apply(fix_sign, axis=1)

            with engine.begin() as conn:
                for _, r in std.iterrows():
                    try:
                        conn.exec_driver_sql(
                            "INSERT INTO transactions (source_file, transaction_id, date, description, amount, currency, category, hash_key, raw_date_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (r.get("source_file"), r.get("transaction_id"),
                             pd.Timestamp(r["date"]).to_pydatetime(),
                             r.get("description"), float(r["amount"]), r.get("currency"),
                             r.get("category"), r.get("hash_key"), r.get("raw_date_text"))
                        )
                        inserted += 1
                    except SQLAlchemyError:
                        pass
            good += 1

        flash_here(note_import, "success", f"Imported {inserted} rows from {good} file(s), {bad} failed.")
        add_toast("success", f"Imported {inserted} rows from {good} file(s), {bad} failed.", key="import_done", persist=6)
        st.session_state["refresh_transactions"] = True

# ================== Settings ==================
with tab_settings:
    st.header("Settings")
    t1, t2, t3, t4 = st.tabs(["Rule Learning", "Rules (friendly)", "Raw files (YAML)", "Backup / Reset"])

    # --- Rule Learning ---
    with t1:
        with engine.begin() as conn:
            df = pd.read_sql(
                "SELECT description, category, SUM(amount) as total, COUNT(*) as n FROM transactions GROUP BY description, category ORDER BY n DESC",
                conn,
            )
        if df.empty:
            st.info("No data yet.")
        else:
            st.dataframe(df.sort_values(["category","n"], ascending=[True, False]).head(100), use_container_width=True)

        st.markdown("**Add a learning rule**")
        c1,c2,c3,c4 = st.columns([1,1,2,1])
        with c1: new_cat = st.text_input("Category (existing or new)", key="rl_cat")
        with c2: prio = st.number_input("Priority", value=0, step=1, key="rl_prio")
        with c3: kw = st.text_input("Keyword to match", key="rl_kw")
        with c4: neg = st.text_input("Negative keyword (optional)", key="rl_neg")

        note_rl = st.empty()
        if st.button("Save learning rule", key="rl_save"):
            if new_cat and kw:
                data = load_yaml(CATS_FILE, {"rules": []})
                rule = next((r for r in data["rules"] if r.get("category")==new_cat), None)
                if rule:
                    rule.setdefault("keywords", [])
                    if kw not in rule["keywords"]: rule["keywords"].append(kw)
                    rule.setdefault("negative_keywords", [])
                    if neg and neg not in rule["negative_keywords"]: rule["negative_keywords"].append(neg)
                    rule["priority"] = int(prio)
                else:
                    data["rules"].append({"category":new_cat, "priority":int(prio),
                                          "keywords":[kw], "negative_keywords":[neg] if neg else []})
                save_yaml(CATS_FILE, data)

                if new_cat not in settings.get("category_colors", {}):
                    settings.setdefault("category_colors", {})[new_cat] = random_pastel_hex()
                    settings["category_colors"] = sanitize_category_colors(settings["category_colors"])
                    save_yaml(SETTINGS_FILE, settings)

                changed = recategorize_all_transactions(load_yaml(CATS_FILE, {"rules": []}), aliases, settings)
                flash_here(note_rl, "success", f"Rule saved. Re-categorized {changed} row(s).")
                add_toast("success", f"Rule saved. Re-categorized {changed} row(s).", key="rl_saved", persist=6)
                st.session_state["refresh_transactions"] = True
            else:
                flash_here(note_rl, "error", "Please provide both a Category and a Keyword.")
                add_toast("error", "Please provide both a Category and a Keyword.", key="rl_missing")

    # --- Rules (friendly compact) ---
    with t2:
        st.markdown("**Categories & Rules (compact editor)**")
        data_rules = load_yaml(CATS_FILE, {"rules": []})
        rows = []
        for r in data_rules["rules"]:
            rows.append({
                "category": r.get("category",""),
                "priority": int(r.get("priority",0)),
                "regex": bool(r.get("regex", False)),
                "keywords": list_to_csv(r.get("keywords", [])),
                "negative_keywords": list_to_csv(r.get("negative_keywords", [])),
            })
        df_rules = pd.DataFrame(rows)
        edited = st.data_editor(
            df_rules, num_rows="dynamic", use_container_width=True,
            column_config={
                "category": "Category",
                "priority": st.column_config.NumberColumn("Priority", step=1),
                "regex": "Use regex",
                "keywords": "Keywords (comma-separated)",
                "negative_keywords": "Negative keywords (comma-separated)",
            },
            key="rules_compact_editor"
        )
        note_rules = st.empty()
        if st.button("Save rules", key="rules_compact_save"):
            out = {"rules":[]}
            for _, r in edited.fillna({"category":"", "keywords":"", "negative_keywords":"", "priority":0}).iterrows():
                if not str(r["category"]).strip(): continue
                out["rules"].append({
                    "category": str(r["category"]).strip(),
                    "priority": int(r["priority"]),
                    "regex": bool(r["regex"]),
                    "keywords": csv_to_list(r["keywords"]),
                    "negative_keywords": csv_to_list(r["negative_keywords"]),
                })
            save_yaml(CATS_FILE, out)
            changed = recategorize_all_transactions(load_yaml(CATS_FILE, {"rules": []}), aliases, settings)
            flash_here(note_rules, "success", f"Rules saved. Re-categorized {changed} row(s).")
            add_toast("success", f"Rules saved. Re-categorized {changed} row(s).", key="rules_saved")
            st.session_state["refresh_transactions"] = True

    # --- Raw YAML editors ---
    with t3:
        cats_text = CATS_FILE.read_text(encoding="utf-8") if CATS_FILE.exists() else ""
        recur_text = RECUR_FILE.read_text(encoding="utf-8") if RECUR_FILE.exists() else ""
        budgets_text = BUDGETS_FILE.read_text(encoding="utf-8") if BUDGETS_FILE.exists() else ""
        aliases_text = ALIASES_FILE.read_text(encoding="utf-8") if ALIASES_FILE.exists() else ""
        settings_text = SETTINGS_FILE.read_text(encoding="utf-8") if SETTINGS_FILE.exists() else ""
        with st.expander("categories.yaml", expanded=False): cats_edit = st.text_area("Categories", value=cats_text, height=180)
        with st.expander("recurring.yaml", expanded=False): recur_edit = st.text_area("Recurring", value=recur_text, height=140)
        with st.expander("budgets.yaml", expanded=False): budgets_edit = st.text_area("Budgets", value=budgets_text, height=140)
        with st.expander("aliases.yaml", expanded=False): aliases_edit = st.text_area("Aliases", value=aliases_text, height=120)
        with st.expander("settings.yaml", expanded=False): settings_edit = st.text_area("Settings", value=settings_text, height=140)
        note_yaml = st.empty()
        if st.button("Save all raw YAML files", key="save_all_yaml"):
            try:
                save_yaml(CATS_FILE, yaml.safe_load(cats_edit) if cats_edit.strip() else {"rules": []})
                save_yaml(RECUR_FILE, yaml.safe_load(recur_edit) if recur_edit.strip() else {"bills": []})
                save_yaml(BUDGETS_FILE, yaml.safe_load(budgets_edit) if budgets_edit.strip() else {"budgets": []})
                save_yaml(ALIASES_FILE, yaml.safe_load(aliases_edit) if aliases_edit.strip() else {"aliases": []})
                new_settings = yaml.safe_load(settings_edit) if settings_edit.strip() else settings
                if new_settings is not None:
                    new_settings["category_colors"] = sanitize_category_colors(new_settings.get("category_colors", {}))
                    save_yaml(SETTINGS_FILE, new_settings)
                changed = recategorize_all_transactions(load_yaml(CATS_FILE, {"rules": []}), aliases, load_yaml(SETTINGS_FILE, settings))
                flash_here(note_yaml, "success", f"Saved all YAML files. Re-categorized {changed} row(s).")
                add_toast("success", f"Saved all YAML files. Re-categorized {changed} row(s).", key="yaml_saved", persist=6)
                st.session_state["refresh_transactions"] = True
            except yaml.YAMLError as e:
                flash_here(note_yaml, "error", f"YAML error: {e}")
                add_toast("error", f"YAML error: {e}", key="yaml_error")

    # --- Backup & Reset (transactions only) ---
    with t4:
        st.subheader("Backup")
        note_backup = st.empty()
        if st.button("Create backup zip", key="backup_zip"):
            name = auto_backup()
            if name:
                flash_here(note_backup, "success", f"Created {name} in backups/")
                add_toast("success", f"Created {name} in backups/")
            else:
                flash_here(note_backup, "error", "Backup failed.")
                add_toast("error", "Backup failed.", key="backup_failed")

        st.markdown("---")
        st.markdown("### Reset transactions (keep presets, rules, budgets & backups)")
        st.info("This deletes **all rows** in the transactions table only.")
        confirm = st.text_input("Type RESET to enable the button", key="reset_confirm_tx")

        note_reset = st.empty()
        if st.button("Delete ALL transactions", disabled=(confirm.strip()!="RESET"), key="reset_tx_only"):
            try:
                with engine.begin() as conn:
                    conn.exec_driver_sql("DELETE FROM transactions;")
                with engine.execution_options(isolation_level="AUTOCOMMIT").connect() as conn:
                    conn.exec_driver_sql("VACUUM")
                flash_here(note_reset, "success", "All transactions deleted. Presets, rules, budgets and backups were not touched.")
                add_toast("success", "All transactions deleted. Presets, rules, budgets and backups were not touched.", key="reset_ok", persist=6)
                st.session_state["refresh_transactions"] = True
            except Exception as e:
                flash_here(note_reset, "error", f"Reset failed: {e}")
                add_toast("error", f"Reset failed: {e}", key="reset_err", persist=6)

# ================== Help ==================
with tab_help:
    st.header("Help & Guide")
    def show_md(p: str):
        pp = HELP_DIR / p
        st.markdown(pp.read_text(encoding="utf-8") if pp.exists() else f"{p} coming soon.")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Dashboard"); show_md("dashboard.md")
        st.markdown("### Upload & Map"); show_md("upload.md")
    with c2:
        st.markdown("### Transactions"); show_md("transactions.md")
        st.markdown("### Upcoming & Recurring"); show_md("upcoming.md")
    st.markdown("### Settings"); show_md("settings.md")
