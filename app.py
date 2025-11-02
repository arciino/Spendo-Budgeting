# --- Budget App v2.6.2 (PublicB) ---
import re
import io
import zipfile
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from dateutil import parser as du
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# ---------- Paths / engine ----------
APP_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
DB_PATH = APP_DIR / "budget.db"
RULES_DIR = APP_DIR / "rules"
CATS_FILE = RULES_DIR / "categories.yaml"
RECUR_FILE = RULES_DIR / "recurring.yaml"
BUDGETS_FILE = RULES_DIR / "budgets.yaml"
ALIASES_FILE = RULES_DIR / "aliases.yaml"
PRESETS_FILE = RULES_DIR / "bank_presets.yaml"
SETTINGS_FILE = RULES_DIR / "settings.yaml"
BACKUPS_DIR = APP_DIR / "backups"
HELP_DIR = APP_DIR / "help"

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
st.set_page_config(page_title="Spendo", page_icon="💸", layout="wide")

# ---------- Utilities ----------
def load_yaml(path: Path, fallback: dict):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(fallback, f, sort_keys=False, allow_unicode=True)
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or fallback
        except yaml.YAMLError:
            return fallback

def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

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

def normalize_amount(x):
    if pd.isna(x): return None
    s = (str(x).strip()
         .replace("CHF","").replace("chf","")
         .replace(" ","").replace("'","")
         .replace(",",".").replace("+",""))
    try: return float(s)
    except: return None

def parse_date(s, *, dayfirst=True, yearfirst=False, fmt=None):
    if s is None or (isinstance(s, float) and pd.isna(s)): return None
    s = str(s).strip()
    if not s: return None
    if fmt:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            return None
    try:
        return du.parse(s, dayfirst=dayfirst, yearfirst=yearfirst).date()
    except Exception:
        return None

def apply_aliases(desc: str, aliases: dict) -> str:
    alias_map = {a.get("match","").lower(): a.get("alias","") for a in aliases.get("aliases",[]) if a.get("match")}
    t = str(desc); low = t.lower()
    for k,v in alias_map.items():
        if k in low: return v if v else t
    return t

# Boundary-aware + priority categorizer with `_match_reason`
def apply_categories(df, cats, aliases):
    def _norm(s: str) -> str:
        t = ("" if s is None else str(s)).casefold()
        t = re.sub(r"\s+", " ", t).strip()
        return t

    if df.empty:
        return df

    df["description"] = df["description"].apply(lambda s: apply_aliases(s, aliases))

    compiled = []
    for r in cats.get("rules", []):
        cat = r.get("category", "Other")
        pr  = int(r.get("priority", 0))
        kws = [str(k) for k in r.get("keywords", []) if str(k).strip()]
        patterns = [re.compile(rf"\b{re.escape(k.casefold())}\b") for k in kws]
        if patterns:
            compiled.append({
                "category": cat,
                "priority": pr,
                "patterns": patterns,
                "keywords": [k.casefold() for k in kws],
            })

    compiled.sort(key=lambda x: x["priority"], reverse=True)

    reasons, cats_out = [], []
    for desc in df["description"].astype(str):
        text = _norm(desc)
        best = ("Other", -1, -1, "")  # (cat, prio, kw_len, kw_text)
        for r in compiled:
            for pat, kw in zip(r["patterns"], r["keywords"]):
                if pat.search(text):
                    score = (r["priority"], len(kw))
                    if score > (best[1], best[2]):
                        best = (r["category"], r["priority"], len(kw), kw)
                    break
        cats_out.append(best[0])
        reasons.append(best[3] or "(no match)")
    df["category"] = cats_out
    df["_match_reason"] = reasons
    return df

def month_bounds(d: date):
    first = d.replace(day=1)
    last = (d.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
    return first, last

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
    except:
        return None

def color_for(cat, settings):
    return settings.get("category_colors", {}).get(cat, "#95a5a6")

def load_presets():
    return load_yaml(PRESETS_FILE, {"presets": []})

# ---------- Load rules / settings ----------
ensure_tables()
cats     = load_yaml(CATS_FILE, {"rules": []})
recur    = load_yaml(RECUR_FILE, {"bills": []})
budgets  = load_yaml(BUDGETS_FILE, {"budgets": []})
aliases  = load_yaml(ALIASES_FILE, {"aliases": []})
settings = load_yaml(SETTINGS_FILE, {
    "auto_sign_detection": True,
    "category_signs": {"Income": 1, "Other": -1},
    "category_colors": {},
    "theme": "dark",
    "date_parse": {"dayfirst": True, "yearfirst": False},
})

# ---------- Theme (top-right) ----------
theme = settings.get("theme","dark")
plotly_theme = "plotly_dark" if theme=="dark" else "plotly_white"
top = st.columns([0.9, 0.1], vertical_alignment="center")
with top[1]:
    if st.button("🌙" if theme == "dark" else "☀️", help="Toggle theme", key="theme_icon"):
        settings["theme"] = "light" if theme == "dark" else "dark"
        save_yaml(SETTINGS_FILE, settings)
        st.rerun()

# ---------- Tabs ----------
tabs = st.tabs(["Dashboard","Transactions","Upcoming & Recurring","Upload & Map","Settings","Help"])

# ================== Dashboard ==================
with tabs[0]:
    st.header("Dashboard")
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
    if df.empty:
        st.info("No data yet. Go to **Upload & Map** to import a CSV.")
    else:
        view = st.radio("View", ["Net","Expenses","Income"], horizontal=True)
        today = date.today(); first, last = month_bounds(today)
        cur = df[(df["date"]>=pd.Timestamp(first)) & (df["date"]<=pd.Timestamp(last))].copy()

        last6 = (df[df["date"]<pd.Timestamp(first)]
                 .assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
                 .groupby("month", as_index=False)["amount"].sum()
                 .sort_values("month").tail(6))
        avg6 = last6["amount"].mean() if not last6.empty else 0.0
        income = cur[cur["amount"]>0]["amount"].sum()
        expense = cur[cur["amount"]<0]["amount"].sum()
        net = income + expense
        a,b,c,d = st.columns(4)
        a.metric("Income (this month)", f"CHF {income:,.2f}")
        b.metric("Expenses (this month)", f"CHF {abs(expense):,.2f}")
        c.metric("Net (this month)", f"CHF {net:,.2f}")
        d.metric("Avg net (last 6 mo)", f"CHF {avg6:,.2f}", delta=f"{net-avg6:+.2f} vs avg")

        base = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        if view=="Net":
            series = base.groupby("month", as_index=False)["amount"].sum()
            fig = px.bar(series.tail(18), x="month", y="amount", title="Monthly Net")
        elif view=="Expenses":
            series = base.assign(expense=(-base["amount"].clip(upper=0))).groupby("month", as_index=False)["expense"].sum()
            fig = px.bar(series.tail(18), x="month", y="expense", title="Monthly Expenses")
        else:
            series = base.assign(income=(base["amount"].clip(lower=0))).groupby("month", as_index=False)["income"].sum()
            fig = px.bar(series.tail(18), x="month", y="income", title="Monthly Income")
        fig.update_layout(hovermode="x unified", template=plotly_theme)
        st.plotly_chart(fig, use_container_width=True)

        exp = cur[cur["amount"]<0].copy()
        if not exp.empty:
            cat = (-exp.groupby("category")["amount"].sum()).reset_index().sort_values("amount", ascending=False)
            cat["color"] = cat["category"].apply(lambda c: color_for(c, settings))
            fig2 = px.pie(cat, values="amount", names="category", hole=0.45,
                          title="Spending by category (this month)",
                          color="category", color_discrete_sequence=cat["color"])
            fig2.update_layout(template=plotly_theme)
            st.plotly_chart(fig2, use_container_width=True)
            drill = st.selectbox("Drill down category", options=cat["category"])
            dd = exp[exp["category"]==drill].sort_values("date", ascending=False)
            st.dataframe(dd[["date","description","amount","currency","source_file"]],
                         use_container_width=True, height=300)
        else:
            st.info("No expenses yet this month.")

# ================== Transactions ==================
with tabs[1]:
    st.header("Transactions")
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM transactions ORDER BY date DESC", conn, parse_dates=["date"])
    if df.empty:
        st.info("No data yet.")
    else:
        cols = st.columns([1,1,1,1,0.6])
        with cols[0]:
            start = st.date_input("Start date", value=df["date"].min().date())
        with cols[1]:
            end = st.date_input("End date", value=df["date"].max().date())
        with cols[2]:
            category = st.selectbox("Category", ["(all)"] + sorted(df["category"].dropna().unique().tolist()))
        with cols[3]:
            search = st.text_input("Search")
        with cols[4]:
            with st.popover("⚙️ Columns"):
                show_cols = st.multiselect(
                    "Visible columns",
                    ["date","description","amount","currency","category","source_file","_match_reason"],
                    default=["date","description","amount","category"]
                )

        mask = (df["date"].dt.date>=start) & (df["date"].dt.date<=end)
        if category!="(all)": mask &= (df["category"]==category)
        if search: mask &= df["description"].str.contains(search, case=False, na=False)
        fdf = df[mask].copy()
        if "date" in fdf.columns:
            fdf["date"] = fdf["date"].dt.strftime("%d.%m.%Y")

        edit_mode = st.toggle("Edit mode", value=False, help="Edit description/category then Confirm")
        editable_cols = ["description","category"]
        display_cols = show_cols

        if edit_mode:
            if "id" not in fdf.columns:
                st.warning("Editing needs 'id' in the table (present by default).")
            else:
                edit_df = fdf[["id"] + list(dict.fromkeys(display_cols + editable_cols))].copy()
                st.caption("Only 'description' and 'category' are editable here.")
                edited = st.data_editor(
                    edit_df,
                    use_container_width=True,
                    num_rows="fixed",
                    disabled=[c for c in edit_df.columns if c not in editable_cols and c != "id"],
                    hide_index=True
                )
                if st.button("Confirm changes", type="primary"):
                    diffs = []
                    orig = edit_df.set_index("id"); new = edited.set_index("id")
                    for col in editable_cols:
                        changed_ids = orig.index[orig[col] != new[col]]
                        for _id in changed_ids:
                            diffs.append((_id, col, new.loc[_id, col]))
                    if not diffs:
                        st.info("No changes detected.")
                    else:
                        n = 0
                        with engine.begin() as conn:
                            for _id, col, val in diffs:
                                if col == "category" and (val is None or str(val).strip() == ""):
                                    val = "Other"
                                conn.exec_driver_sql(f"UPDATE transactions SET {col}=? WHERE id=?", (str(val), int(_id)))
                                n += 1
                        st.success(f"Applied {n} field updates."); st.rerun()
        else:
            def style_rows(row):
                return [f"background-color: {color_for(row.get('category','Other'), settings)}; color: white;" for _ in row]
            st.dataframe(fdf[display_cols].style.apply(style_rows, axis=1), use_container_width=True)

        st.download_button("Export filtered to CSV",
                           fdf[display_cols].to_csv(index=False).encode("utf-8"),
                           file_name="filtered_transactions.csv", mime="text/csv")

        # Inspector
        with st.expander("🔎 Why did this get its category?"):
            q = st.text_input("Paste a description to test", value="")
            if q:
                tmp = pd.DataFrame([{"description": q, "amount": 0}])
                out = apply_categories(tmp.copy(), cats, aliases)
                st.write("Category →", out.loc[0, "category"])
                if "_match_reason" in out.columns:
                    st.caption(f"Matched keyword: {out.loc[0, '_match_reason']}")

# ================== Upcoming & Recurring ==================
with tabs[2]:
    st.header("Upcoming & Recurring Payments")
    with engine.begin() as conn:
        tdf = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
    suggestions = []
    if not tdf.empty:
        tdf["ym"] = tdf["date"].dt.to_period("M")
        grp = tdf.groupby(["description","category"])
        for (desc, cat), sub in grp:
            months = sorted(sub["ym"].unique())
            if len(months) >= 3:
                med = sub["amount"].abs().median()
                if med and (abs(sub["amount"].abs()-med) <= 0.05*med).mean() >= 0.7:
                    suggestions.append({"name": str(desc)[:48], "category": cat, "amount": round(float(med),2)})
    if suggestions:
        st.subheader("Suggested recurring bills")
        st.dataframe(pd.DataFrame(suggestions).drop_duplicates(subset=["name","category"]), use_container_width=True)

    recur = load_yaml(RECUR_FILE, {"bills": []})
    bills = recur.get("bills", [])
    today = date.today()
    if not bills:
        st.info("No recurring bills configured. Add entries in **Settings → Rules → recurring.yaml**.")
    else:
        rows = []
        for b in bills:
            d = int(b.get("day",1))
            last_day = (today.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
            due = date(today.year, today.month, min(d, last_day.day))
            if due < today:
                y, m = (today.year+1,1) if today.month==12 else (today.year, today.month+1)
                last_next = (date(y, m, 28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
                due = date(y, m, min(d, last_next.day))
            rows.append({"name": b.get("name",""), "category": b.get("category",""), "usual_day": d, "next_due": due, "amount_estimate": float(b.get("amount",0)), "notes": b.get("notes","")})
        st.dataframe(pd.DataFrame(rows).sort_values("next_due"), use_container_width=True)

# ================== Upload & Map (Preview -> Import) ==================
with tabs[3]:
    st.header("Upload & Map")

    cols_top = st.columns([1, 1, 2])
    with cols_top[0]:
        if st.button("🔄 Reload presets"):
            st.rerun()

    presets = load_presets()
    preset_names = ["(auto-detect per file)"] + [p["name"] for p in presets.get("presets", [])]

    c1, c2 = st.columns([2, 1])
    with c1:
        preset_name = st.selectbox("Preset", preset_names, index=0,
                                   help="Select a preset or keep auto-detect per file.")
    with c2:
        delim = st.selectbox("Delimiter", ["(auto)", ",", ";", "\\t"])

    uploads = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True)

    map1, map2, map3 = st.columns(3)
    with map1:
        date_col = st.text_input("Date column", value="date")
        amount_col = st.text_input("Amount column", value="amount")
    with map2:
        desc_col = st.text_input("Description column", value="description")
        currency_col = st.text_input("Currency column (optional)", value="currency")
    with map3:
        id_col = st.text_input("Transaction ID (optional)", value="transaction_id")
        st.caption("Save a preset later so mapping is automatic.")

    st.divider()

    def try_read_bytes(raw, delim_choice):
        if delim_choice != "(auto)":
            sep = "," if delim_choice == "," else (";" if delim_choice == ";" else "\t")
            return pd.read_csv(io.BytesIO(raw), sep=sep)
        for sep in [",", ";", "\t"]:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=sep)
            except Exception:
                pass
        return None

    def detect_preset(presets_obj, filename, headers_lower):
        name = filename.lower()
        for p in presets_obj.get("presets", []):
            fn = [x.lower() for x in p.get("match", {}).get("filename_contains", [])]
            hn = [x.lower() for x in p.get("match", {}).get("header_contains", [])]
            fn_ok = (not fn) or any(x in name for x in fn)
            hn_ok = (not hn) or all(any(h in col for col in headers_lower) for h in hn)
            if fn_ok and hn_ok:
                return p
        return None

    def standardize_one(up, df_in, chosen, settings,
                        date_col, desc_col, amount_col, currency_col, id_col):
        mapping = chosen.get("mapping", {})
        missing = [mapping.get("date"), mapping.get("description"), mapping.get("amount")]
        if any(m not in df_in.columns for m in missing):
            mapping = {
                "date": date_col,
                "description": desc_col,
                "amount": amount_col,
                "currency": currency_col,
                "transaction_id": id_col,
            }

        prefs = settings.get("date_parse", {"dayfirst": True, "yearfirst": False})
        p_dayfirst  = chosen.get("dayfirst",  prefs.get("dayfirst", True))
        p_yearfirst = chosen.get("yearfirst", prefs.get("yearfirst", False))
        p_fmt       = chosen.get("date_format", None)

        std = pd.DataFrame()
        raw_date_series = df_in[mapping.get("date")].astype(str)
        std["raw_date_text"] = raw_date_series
        std["date"] = raw_date_series.apply(
            lambda v: parse_date(v, dayfirst=p_dayfirst, yearfirst=p_yearfirst, fmt=p_fmt)
        )

        std["description"] = df_in[mapping.get("description")].astype(str)
        builder = chosen.get("build_description")
        if builder:
            ok = [c for c in builder if c in df_in.columns]
            if ok:
                std["description"] = df_in[ok].astype(str).agg(" ".join, axis=1).str.strip()
        elif {"MerchantName", "Details"}.issubset(set(df_in.columns)):
            std["description"] = df_in[["MerchantName", "Details"]].astype(str).agg(" ".join, axis=1).str.strip()

        std["amount"] = df_in[mapping.get("amount")].apply(normalize_amount)
        cur_col = mapping.get("currency")
        idc_col = mapping.get("transaction_id")
        std["currency"] = df_in[cur_col] if cur_col and cur_col in df_in.columns else "CHF"
        std["transaction_id"] = df_in[idc_col] if idc_col and idc_col in df_in.columns else ""
        std["source_file"] = up.name

        before = len(std)
        std = std.dropna(subset=["date", "amount"])
        dropped = before - len(std)

        std = apply_categories(std, cats, aliases)
        if settings.get("auto_sign_detection", True):
            cat_signs = settings.get("category_signs", {})
            def fix_sign(row):
                cat = row.get("category","Other"); sign = cat_signs.get(cat, -1)
                amt = abs(float(row["amount"])) if pd.notna(row["amount"]) else row["amount"]
                return amt if sign >= 0 else -abs(amt)
            std["amount"] = std.apply(fix_sign, axis=1)

        std["_date_str"] = pd.to_datetime(std["date"]).dt.strftime("%d.%m.%Y")
        return std, dropped, {"dayfirst": p_dayfirst, "yearfirst": p_yearfirst, "date_format": p_fmt}

    preview_clicked = st.button("👀 Preview imports (no save)", type="secondary", disabled=not uploads)
    if preview_clicked:
        st.session_state["__preview_rows__"] = []
        st.session_state["__preview_meta__"] = []
        bad_files = 0

        for up in uploads:
            up.seek(0)
            raw = up.read()
            df_in = try_read_bytes(raw, delim)
            if df_in is None or df_in.empty:
                bad_files += 1
                st.error(f"Failed to read {up.name}.")
                continue

            headers_lower = [str(c).lower() for c in df_in.columns]
            chosen = None
            if preset_name != "(auto-detect per file)":
                chosen = next((p for p in presets.get("presets", []) if p["name"] == preset_name), None)
            if chosen is None:
                chosen = detect_preset(presets, up.name, headers_lower) or {
                    "mapping": {
                        "date": date_col, "description": desc_col,
                        "amount": amount_col, "currency": currency_col, "transaction_id": id_col,
                    },
                    "delimiter": ",",
                }

            std, dropped, parse_meta = standardize_one(
                up, df_in, chosen, settings, date_col, desc_col, amount_col, currency_col, id_col
            )

            st.markdown(
                f"**Preview:** `{up.name}` — rows: {len(std)} (dropped: {dropped})  \n"
                f"Preset: `{chosen.get('name', '(ad-hoc)')}` | "
                f"date_format: `{parse_meta['date_format']}` | "
                f"dayfirst: `{parse_meta['dayfirst']}` | yearfirst: `{parse_meta['yearfirst']}`"
            )
            if not std.empty:
                # Controls for preview size
                max_preview = len(std)
                col_a, col_b = st.columns([1,1])
                with col_a:
                    show_all = st.checkbox("Show all rows in preview (may be heavy)", value=False, key=f"prev_all_{up.name}")
                with col_b:
                    limit = st.number_input("Rows to show", min_value=10, max_value=max_preview,
                                            value=min(200, max_preview), step=10, key=f"prev_lim_{up.name}")

                preview_df = std[["raw_date_text", "_date_str", "description", "amount", "category"]]
                if not show_all:
                    preview_df = preview_df.head(limit)

                st.dataframe(preview_df, use_container_width=True, height=400)
                st.session_state["__preview_rows__"].append(std)
                st.session_state["__preview_meta__"].append({"file": up.name, **parse_meta})

        if bad_files == 0 and not st.session_state["__preview_rows__"]:
            st.info("Nothing to preview. Check your delimiter or mapping.")

    import_clicked = st.button("⬇️ Import above preview to database", type="primary")
    if import_clicked:
        rows_list = st.session_state.get("__preview_rows__", [])
        if not rows_list:
            st.warning("No preview available. Click 'Preview imports (no save)' first.")
        else:
            z = auto_backup()
            if z:
                st.info(f"Backup created: backups/{z}")
            all_rows = 0
            inserted = 0
            try:
                with engine.begin() as conn:
                    for std in rows_list:
                        std["hash_key"] = std.apply(
                            lambda r: f"{r.get('source_file','')}|{(r.get('date') or '')}|"
                                      f"{round(float(r.get('amount',0.0)),2)}|{r.get('description','')}".lower(),
                            axis=1,
                        )
                        all_rows += len(std)
                        for _, row in std.iterrows():
                            try:
                                conn.exec_driver_sql(
                                    """INSERT INTO transactions
                                       (source_file, transaction_id, date, description, amount, currency, category, hash_key, raw_date_text)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        row["source_file"], row["transaction_id"],
                                        pd.Timestamp(row["date"]).to_pydatetime(),
                                        row["description"], float(row["amount"]), str(row["currency"]),
                                        row["category"], row["hash_key"], row.get("raw_date_text", ""),
                                    ),
                                )
                                inserted += 1
                            except SQLAlchemyError:
                                pass
                st.success(f"Imported {inserted}/{all_rows} rows. Go to Transactions to review.")
                st.session_state["__preview_rows__"] = []
                st.session_state["__preview_meta__"] = []
            except Exception as e:
                st.error(f"Error during import: {e}")

    with st.expander("Preset debug", expanded=False):
        st.write("Loaded from:", PRESETS_FILE)
        st.write("Found presets:", [p.get("name") for p in presets.get("presets", [])])

# ================== Settings ==================
with tabs[4]:
    st.header("Settings")
    t1,t2,t3 = st.tabs(["Rule Learning","Rules","Backup"])

    # Rule Learning
    with t1:
        st.subheader("Rule Learning")
        with engine.begin() as conn:
            df = pd.read_sql(
                "SELECT description, category, SUM(amount) as total, COUNT(*) as n "
                "FROM transactions GROUP BY description, category ORDER BY n DESC", conn
            )
        if df.empty:
            st.info("No data yet.")
        else:
            df["prio"] = (df["category"]=="Other").astype(int)
            top = df.sort_values(["prio","n"], ascending=[False, False]).head(50)
            st.dataframe(top, use_container_width=True)
            existing = sorted({r.get("category") for r in cats.get("rules", []) if r.get("category")})
            chosen_desc = st.text_input("Description to learn")
            col1,col2 = st.columns(2)
            with col1: target_cat = st.selectbox("Category", options=["(new)…"]+existing)
            with col2: new_cat = st.text_input("If new, type name")
            kw = st.text_input("Keyword to match", value="")
            if st.button("Add rule"):
                keyword = kw.strip() if kw.strip() else chosen_desc.strip()
                category = new_cat.strip() if target_cat=="(new)…" else target_cat
                if keyword and category:
                    data = load_yaml(CATS_FILE, {"rules": []})
                    found=False
                    for r in data["rules"]:
                        if r.get("category")==category:
                            r.setdefault("keywords", [])
                            if keyword not in r["keywords"]: r["keywords"].append(keyword)
                            found=True; break
                    if not found: data["rules"].append({"category": category, "keywords": [keyword]})
                    save_yaml(CATS_FILE, data); st.success(f"Saved rule: '{keyword}' → {category}")

    # Rules + maintenance
    with t2:
        st.subheader("Rules")
        cats_text     = Path(CATS_FILE).read_text(encoding="utf-8") if Path(CATS_FILE).exists() else ""
        recur_text    = Path(RECUR_FILE).read_text(encoding="utf-8") if Path(RECUR_FILE).exists() else ""
        budgets_text  = Path(BUDGETS_FILE).read_text(encoding="utf-8") if Path(BUDGETS_FILE).exists() else ""
        aliases_text  = Path(ALIASES_FILE).read_text(encoding="utf-8") if Path(ALIASES_FILE).exists() else ""
        settings_text = Path(SETTINGS_FILE).read_text(encoding="utf-8") if Path(SETTINGS_FILE).exists() else ""

        st.write("**categories.yaml**")
        cats_edit = st.text_area("categories.yaml", value=cats_text, height=200)
        st.write("**recurring.yaml**")
        recur_edit = st.text_area("recurring.yaml", value=recur_text, height=160)
        st.write("**budgets.yaml**")
        budgets_edit = st.text_area("budgets.yaml", value=budgets_text, height=160)
        st.write("**aliases.yaml**")
        aliases_edit = st.text_area("aliases.yaml", value=aliases_text, height=140)
        st.write("**settings.yaml**")
        settings_edit = st.text_area("settings.yaml", value=settings_text, height=180)

        if st.button("Save all rule files"):
            try:
                save_yaml(CATS_FILE, yaml.safe_load(cats_edit) if cats_edit.strip() else {"rules": []})
                save_yaml(RECUR_FILE, yaml.safe_load(recur_edit) if recur_edit.strip() else {"bills": []})
                save_yaml(BUDGETS_FILE, yaml.safe_load(budgets_edit) if budgets_edit.strip() else {"budgets": []})
                save_yaml(ALIASES_FILE, yaml.safe_load(aliases_edit) if aliases_edit.strip() else {"aliases": []})
                save_yaml(SETTINGS_FILE, yaml.safe_load(settings_edit) if settings_edit.strip() else settings)
                st.success("Saved all rule files.")
            except yaml.YAMLError as e:
                st.error(f"YAML error: {e}")

        st.divider()
        if st.button("Preview re-categorization"):
            with engine.begin() as conn:
                df_all = pd.read_sql("SELECT id, description, category FROM transactions", conn)
            if df_all.empty:
                st.info("No transactions in database.")
            else:
                tmp = df_all.copy()
                tmp = apply_categories(tmp.rename(columns={"description":"description"}), cats, aliases)
                merged = df_all.merge(tmp[["id","category"]].rename(columns={"category":"new_category"}), on="id", how="left")
                diff = merged[merged["category"] != merged["new_category"]]
                st.write(f"Changes to apply: {len(diff)}")
                if not diff.empty:
                    st.dataframe(diff.head(200), use_container_width=True)
                st.session_state["recat_preview"] = diff.to_dict("records")

        if st.button("Apply re-categorization"):
            changes = st.session_state.get("recat_preview", [])
            if not changes:
                st.warning("No preview or no changes. Click 'Preview re-categorization' first.")
            else:
                with engine.begin() as conn:
                    for r in changes:
                        conn.exec_driver_sql("UPDATE transactions SET category=? WHERE id=?", (r["new_category"], int(r["id"])))
                st.success(f"Applied {len(changes)} updates.")

        st.divider()
        if st.button("Re-categorize EVERYTHING now (no preview)"):
            with engine.begin() as conn:
                df_all = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["date"])
            if df_all.empty:
                st.info("No transactions in database.")
            else:
                df_all = apply_categories(df_all, cats, aliases)
                if settings.get("auto_sign_detection", True):
                    cat_signs = settings.get("category_signs", {})
                    def fix_sign(row):
                        cat = row.get("category","Other"); sign = cat_signs.get(cat, -1)
                        amt = abs(float(row["amount"])) if pd.notna(row["amount"]) else row["amount"]
                        return amt if sign >= 0 else -abs(amt)
                    df_all["amount"] = df_all.apply(fix_sign, axis=1)
                with engine.begin() as conn:
                    for _, r in df_all.iterrows():
                        conn.exec_driver_sql(
                            "UPDATE transactions SET category=?, amount=? WHERE id=?",
                            (r["category"], float(r["amount"]), int(r["id"]))
                        )
                st.success("All rows re-categorized and signs re-applied.")

        st.divider()
        st.subheader("Date utilities")
        if st.button("Fix ambiguous dates (re-parse rows in the future as day-first)"):
            prefs = settings.get("date_parse", {"dayfirst": True, "yearfirst": False})
            with engine.begin() as conn:
                df_all = pd.read_sql("SELECT id, date, raw_date_text FROM transactions", conn, parse_dates=["date"])
            if df_all.empty:
                st.info("No transactions in database.")
            else:
                today = date.today()
                cand = df_all[(df_all["date"].dt.date > today) & df_all["raw_date_text"].notna()].copy()
                cand["new_date"] = cand["raw_date_text"].astype(str).apply(
                    lambda v: parse_date(v, dayfirst=prefs.get("dayfirst", True), yearfirst=prefs.get("yearfirst", False))
                )
                cand = cand[cand["new_date"].notna()]
                if cand.empty:
                    st.success("No ambiguous future dates to fix.")
                else:
                    with engine.begin() as conn:
                        for _, r in cand.iterrows():
                            conn.exec_driver_sql(
                                "UPDATE transactions SET date=? WHERE id=?",
                                (pd.Timestamp(r["new_date"]).to_pydatetime(), int(r["id"]))
                            )
                    st.success(f"Re-parsed and corrected {len(cand)} rows. Refresh the Transactions tab.")
                    st.rerun()

    # Backup
    with t3:
        st.subheader("Backup")
        rows = 0
        try:
            with engine.begin() as conn:
                r = conn.exec_driver_sql("SELECT COUNT(*) FROM transactions").fetchone()
                rows = r[0] if r else 0
        except:
            pass
        st.write(f"DB rows: {rows}")
        last_backup = sorted(BACKUPS_DIR.glob("backup_*.zip"))[-1].name if list(BACKUPS_DIR.glob("backup_*.zip")) else "—"
        st.write(f"Last backup: {last_backup}")
        if st.button("Create backup now"):
            z = auto_backup()
            st.success(f"Backup created: backups/{z}" if z else "Backup failed")

# ================== Help ==================
with tabs[5]:
    st.header("Help & Guide")
    def show(p):
        pp = (HELP_DIR/p)
        st.markdown(pp.read_text(encoding="utf-8") if pp.exists() else f"{p} coming soon.")
    st.subheader("Pages")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### Dashboard");    show("dashboard.md")
        st.markdown("### Upload & Map"); show("upload.md")
    with c2:
        st.markdown("### Transactions"); show("transactions.md")
        st.markdown("### Upcoming & Recurring"); show("upcoming.md")
    st.markdown("### Settings");         show("settings.md")
