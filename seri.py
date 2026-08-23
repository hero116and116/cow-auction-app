import streamlit as st
import pandas as pd
import json
import os
import textwrap
import requests
from google import genai
from google.genai import types

# --- 牛のピクトグラム画像（base64埋め込み・単一ファイルで完結させるため） ---
COW_ICON_B64 = "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"

st.set_page_config(page_title="かう(セリのボーダー計算、結果保存アプリ)", page_icon="🐄", layout="centered")


# --- カスタムCSS（専用テンキー・牛カードデザイン／モックアップ準拠） ---
st.markdown("""
<style>
    /* 全体フォント・余白調整 */
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 480px; }

    /* タブバーが画面上部の固定ツールバーと重なって見づらくなる問題を解消
       （sticky指定を外し、背景を不透明にして通常のスクロール要素にする） */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        position: static !important;
        background-color: #ffffff !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid="stTabs"] { position: static !important; }
    /* タブのラベル同士が重なって表示される問題を解消
       （幅を確保しつつ、はみ出す分は横スクロールで見えるようにする） */
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        overflow: visible !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        white-space: nowrap !important;
    }

    /* 画面全体を囲む1枚のカード（モックアップの外枠） */
    .screen-card {
        border: 2px solid #1e293b;
        border-radius: 4px;
        margin-bottom: 16px;
        overflow: hidden;
        background-color: #ffffff;
    }
    .card-top {
        padding: 20px 18px 16px 18px;
        text-align: center;
        position: relative;
        min-height: 300px;
        box-sizing: border-box;
    }
    .card-divider {
        border-top: 2px solid #1e293b;
        margin: 0;
    }
    .card-bottom {
        padding: 16px 14px 18px 14px;
    }

    /* 出場番号バッジ */
    .cow-no-label {
        font-size: 16px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 4px;
    }

    /* 牛のピクトグラム */
    .cow-icon-container {
        position: relative;
        display: inline-block;
        width: 170px;
        height: 110px;
        margin: 4px 0;
    }
    .cow-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .cow-number-overlay {
        position: absolute;
        top: 50%;
        left: 52%;
        transform: translate(-50%, -50%);
        color: #ffffff;
        font-size: 24px;
        font-weight: 900;
        text-shadow: 0 0 4px rgba(0,0,0,0.9), 0 0 2px rgba(0,0,0,0.9);
    }

    /* 体重・落札額の大きな入力表示（下線スタイル） */
    .input-display-row {
        display: flex;
        align-items: baseline;
        justify-content: center;
        gap: 8px;
        margin-top: 10px;
    }
    .input-display {
        font-size: 32px;
        font-weight: 800;
        color: #1e293b;
        border-bottom: 2px solid #1e293b;
        display: inline-block;
        min-width: 160px;
        height: 40px;
        line-height: 40px;
        padding: 2px 6px;
        text-align: right;
        white-space: nowrap;
        overflow: hidden;
    }
    .input-unit { font-size: 18px; font-weight: 700; color: #1e293b; }

    .cow-meta {
        margin-top: 10px;
        color: #475569;
        font-size: 14px;
        line-height: 1.6;
        text-align: left;
        display: inline-block;
    }
    .cow-metrics {
        margin-top: 8px;
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
        text-align: left;
    }
    .cow-metrics .profit { color: #059669; font-size: 15px; }
    .cow-metrics .border-price { color: #2563eb; font-size: 16px; }

    /* 購入チェック（右上の小さな四角） */
    .purchase-check-label {
        position: absolute;
        top: 12px;
        right: 14px;
        font-size: 11px;
        font-weight: 700;
        color: #334155;
        text-align: center;
    }

    /* No.X クリックで牛を切り替えるジャンプボタン */
    .st-key-no_jump_w, .st-key-no_jump_p {
        display: flex;
        justify-content: center;
        margin-bottom: 4px;
    }
    .st-key-no_jump_w button, .st-key-no_jump_p button {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #1e293b !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        padding: 4px 14px !important;
    }
    .st-key-no_jump_w button:hover, .st-key-no_jump_p button:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
    }

    /* 購入チェック：テンキー真上の隙間に設置（カードとテンキーをつなぐ帯） */
    .st-key-purchase_check_area_p {
        border: 2px solid #1e293b;
        border-top: none;
        margin-top: -16px;
        padding: 8px 14px;
        background-color: #ffffff;
    }
    .st-key-purchase_check_area_p div[data-testid="stCheckbox"] {
        display: flex;
        justify-content: flex-end;
    }

    /* テンキー全体を囲むコンテナ：カード下段として視覚的につなげる */
    .st-key-numpad_area_w, .st-key-numpad_area_p {
        border: 2px solid #1e293b;
        border-top: none;
        border-radius: 0 0 4px 4px;
        margin-top: -16px;
        padding: 12px 6px 16px 6px;
        background-color: #ffffff;
    }

    /* --- テンキー行のレイアウト（Gemini版に準拠） ---
       flex-grow はここで触らない：st.columns([...]) の比率が
       Streamlit側のinlineスタイルとしてそのまま効くようにするため。
       ここでは「縮められるようにする」ことだけを担当する。 */
    .st-key-numpad_area_w div[data-testid="stHorizontalBlock"],
    .st-key-numpad_area_p div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
    }
    .st-key-numpad_area_w div[data-testid="stColumn"],
    .st-key-numpad_area_p div[data-testid="stColumn"],
    .st-key-numpad_area_w div[data-testid="column"],
    .st-key-numpad_area_p div[data-testid="column"] {
        width: auto !important;
        min-width: 0 !important;
    }
    .st-key-numpad_area_w button,
    .st-key-numpad_area_p button {
        height: 72px !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        padding: 0 !important;
    }
    .st-key-numpad_area_w button:hover,
    .st-key-numpad_area_p button:hover { border-color: #3b82f6 !important; color: #3b82f6 !important; }
    .st-key-numpad_area_w button:focus,
    .st-key-numpad_area_p button:focus,
    .st-key-numpad_area_w button:focus-visible,
    .st-key-numpad_area_p button:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

    /* 決定ボタンだけ強調 */
    .st-key-btn_w_enter button, .st-key-btn_p_enter button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }
    /* クリアボタン */
    .st-key-btn_w_c button, .st-key-btn_p_c button {
        background-color: #f1f5f9 !important;
        color: #dc2626 !important;
    }

    /* 左右ナビゲーションボタン（縦長・モックアップの矢印ボタン） */
    .st-key-prev_w button, .st-key-next_w button,
    .st-key-prev_p button, .st-key-next_p button {
        height: 300px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        margin-top: 0 !important;
    }
    
    /* Streamlit標準の開閉ボタン（>>）はバージョンによって描画が不安定なため
       使用しない。右上のメニュー等ごと非表示にし、代わりに自前ボタンで
       サイドバーの開閉を制御する（後述の st.button を参照）。 */
    [data-testid="stToolbar"],
    [data-testid="stAppToolbar"],
    .stAppToolbar,
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"] button,
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] button[aria-label*="lose"],
    section[data-testid="stSidebar"] button[aria-label*="ollapse"],
    footer,
    div[class*="viewerBadge"],
    iframe[title*="streamlit"] {
        display: none !important;
    }

    /* 自前の「設定を開く」フローティングボタン */
    .st-key-open_sidebar_btn {
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 999999 !important;
    }
    .st-key-open_sidebar_btn button {
        border-radius: 6px !important;
        border: 1px solid #1e293b !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        padding: 4px 10px !important;
    }
    /* サイドバーの開閉はこちらで完全に制御するため、Streamlit内部の
       aria-expanded に連動したスライドアニメーション(transform)や
       幅の縮小を無効化し、常に通常表示のレイアウトに固定する */
    section[data-testid="stSidebar"] {
        transform: none !important;
        min-width: 21rem !important;
        width: 21rem !important;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        transform: none !important;
        width: 21rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- APIキー & kintone設定 ---
GEMINI_API_KEY = "AQ.Ab8RN6KGaI97aQ0liR_8kYw5ALr-SMS8KzDW8cPaMUnlt4veDQ"
KINTONE_DOMAIN = "cattlook.cybozu.com"
KINTONE_APP_ID = "131"
KINTONE_API_TOKEN = "T4aTJyzRN736eaqzzWucxZIIbXy9wYn5YkAnlJsO"

# --- サイドバー開閉状態（Streamlit標準の開閉ボタンに頼らず自前で管理） ---
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False

if not st.session_state.sidebar_open:
    # 閉じているときはサイドバー領域自体を非表示にする
    st.markdown(
        '<style>section[data-testid="stSidebar"]{display:none !important;}</style>',
        unsafe_allow_html=True,
    )
    # 開くための丸ボタンを左上に固定表示
    if st.button("⚙️", key="open_sidebar_btn"):
        st.session_state.sidebar_open = True
        st.rerun()

# --- サイドバー設定 ---
with st.sidebar:
    if st.button("✕ 閉じる", key="close_sidebar_btn", use_container_width=True):
        st.session_state.sidebar_open = False
        st.rerun()
    st.header("⚙️ 共通設定（相場・コスト）")
    carcass_price = st.number_input("枝肉単価 (円/kg)", value=2500, step=50)
    daily_cost = st.number_input("1日あたり育成コスト (円)", value=850, step=10)
    shipment_days = st.number_input("出荷日齢 (日)", value=854, step=1)
    birth_weight = st.number_input("生時体重 (kg)", value=35.0, step=1.0)
    yield_rate = st.number_input("歩留基準 (0.65 = 65%)", value=0.65, step=0.01)

# --- 計算ロジック ---
def calculate_cow_metrics(cow_row):
    try:
        days = float(cow_row.get("日齢", 0))
        weight = float(cow_row.get("体重", 0))
    except (ValueError, TypeError):
        days, weight = 0, 0
    
    if days <= 0 or weight <= birth_weight:
        return {"DG": 0.0, "育成日数": 0, "育成コスト": 0, "予測出荷体重": 0.0, "予測枝肉重量": 0.0, "見込売上": 0, "ボーダー価格": 0}
    
    dg = (weight - birth_weight) / days
    raising_days = max(0, shipment_days - days)
    cost = int(raising_days * daily_cost)
    pred_ship_weight = weight + (dg * raising_days)
    pred_carcass_weight = pred_ship_weight * yield_rate
    sales = int(pred_carcass_weight * carcass_price)
    border_price = max(0, (sales - cost) // 1000)
    
    return {
        "DG": round(dg, 3),
        "育成日数": int(raising_days),
        "育成コスト": cost // 1000,
        "予測出荷体重": round(pred_ship_weight, 1),
        "予測枝肉重量": round(pred_carcass_weight, 1),
        "見込売上": sales // 1000,
        "ボーダー価格": border_price,
    }

# --- 本日の推定平均利益（落札額が判明している牛について、
#     ボーダー価格－実際落札額 を求め、その平均を取る） ---
def calculate_today_avg_profit():
    diffs = []
    for c in st.session_state.cows:
        price = c.get("実際落札額", 0)
        if price and price > 0:
            m = calculate_cow_metrics(c)
            diffs.append(m["ボーダー価格"] - price)
    if not diffs:
        return 0
    return int(round(sum(diffs) / len(diffs)))

# --- 性別正規化 ---
def clean_gender(val):
    s = str(val).strip()
    return "雌" if ("雌" in s or "メス" in s or "めす" in s) else "去"

# --- Gemini 名簿解析 ---
def parse_catalog_file(uploaded_file, key=GEMINI_API_KEY):
    client = genai.Client(api_key=key)
    file_bytes = uploaded_file.getvalue()
    mime_type = "application/pdf" if uploaded_file.name.lower().endswith(".pdf") else "image/jpeg"

    prompt = """
    添付された牛のセリ名簿から各行の情報を抽出し、JSON配列として出力してください。
    キー: No (整数), 性別 (去/雌), 生年月日 (文字列、名簿の表記そのまま。例: R07.11.08), 日齢 (整数), 産次 (整数、無ければ0), 摘要 (文字列), 父 (文字列), 母の父 (文字列), 母の祖父 (文字列), 母の母の祖父 (文字列)
    ※ 体重・落札額は0にしてください。JSON配列のみを出力してください。
    """
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = json.loads(response.text)
    for r in data:
        r["性別"] = clean_gender(r.get("性別", "去"))
    return data

# --- kintone 送信 ---
def send_to_kintone(cows_list):
    url = f"https://{KINTONE_DOMAIN}/k/v1/records.json"
    headers = {"X-Cybozu-API-Token": KINTONE_API_TOKEN, "Content-Type": "application/json"}
    
    records = []
    for c in cows_list:
        weight = float(c.get("体重", 0))
        price = int(c.get("実際落札額", 0))
        if weight > 0 or price > 0:
            status = "購入" if c.get("自社落札", False) else "非購入"
            calc = calculate_cow_metrics(c)
            records.append({
                "出場番号": {"value": int(c["No"])},
                "当日体重": {"value": weight},
                "日齢": {"value": int(c.get("日齢", 0))},
                "産次": {"value": int(c.get("産次", 0))},
                "摘要": {"value": str(c.get("摘要", ""))},
                "性別": {"value": c.get("性別", "去")},
                "父": {"value": str(c.get("父", ""))},
                "母の父": {"value": str(c.get("母の父", ""))},
                "母の祖父": {"value": str(c.get("母の祖父", ""))},
                "母の母の祖父": {"value": str(c.get("母の母の祖父", ""))},
                "落札上限価格": {"value": calc["ボーダー価格"]},
                "実際落札額": {"value": price},
                "購入結果": {"value": status},
            })
    if not records:
        return False, "送信対象のデータがありません。"
    res = requests.post(url, headers=headers, json={"app": KINTONE_APP_ID, "records": records})
    return (True, f"✅ {len(records)} 頭のデータをkintoneに保存しました！") if res.status_code == 200 else (False, f"❌ エラー: {res.text}")

# --- セッションステート初期化 ---
if "cows" not in st.session_state:
    st.session_state.cows = [
        {"No": i, "体重": 0, "実際落札額": 0, "性別": "去", "生年月日": "-", "日齢": 280, "産次": 1, "父": "福勝鶴", "母の父": "美津照重", "母の祖父": "平茂勝", "母の母の祖父": "-", "摘要": "", "自社落札": False}
        for i in range(1, 31)
    ]
if "curr_idx_w" not in st.session_state:
    st.session_state.curr_idx_w = 0
if "curr_idx_p" not in st.session_state:
    st.session_state.curr_idx_p = 0
if "input_buffer" not in st.session_state:
    st.session_state.input_buffer = ""

# --- 牛のピクトグラムヘルパー ---
def get_cow_svg(number_str):
    html = f"""<div class="cow-icon-container">
<img class="cow-img" src="data:image/png;base64,{COW_ICON_B64}" alt="cow"/>
<div class="cow-number-overlay">{number_str}</div>
</div>"""
    return html

# --- メインタブ ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 事前データ自動読み取り", 
    "⚖️ 体重入力（下見）", 
    "🎯 落札価格入力（本番）", 
    "📊 セリ結果一覧"
])

# =========================================================
# 画面1: 事前データ自動読み取り画面
# =========================================================
with tab1:
    st.subheader("📄 セリ名簿の自動読み取り")

    if st.session_state.get("just_parsed_count"):
        st.success(f"✅ 読み取りが完了しました！（{st.session_state.just_parsed_count}頭のデータを読み込みました）")
        st.session_state.just_parsed_count = 0

    uploaded = st.file_uploader("名簿ファイルまたは写真を選択", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded and st.button("🚀 自動読み取り開始", type="primary", use_container_width=True):
        with st.spinner("AIが名簿を解析中..."):
            parsed = parse_catalog_file(uploaded)
            if parsed:
                for r in parsed:
                    r["体重"] = 0
                    r["実際落札額"] = 0
                    r["自社落札"] = False
                st.session_state.cows = parsed
                st.session_state.curr_idx_w = 0
                st.session_state.curr_idx_p = 0
                st.session_state.input_buffer = ""
                st.session_state.just_parsed_count = len(parsed)
                st.toast("読み取りが完了しました！", icon="✅")
                st.rerun()

# =========================================================
# 画面2: 体重入力画面（下見）
# =========================================================
@st.fragment
def render_weight_tab():
    total = len(st.session_state.cows)
    idx = st.session_state.curr_idx_w
    cow = st.session_state.cows[idx]
    
    # 1. No.Xをクリックすると番号を直接入力して任意の牛へ移動できる
    with st.container(key="no_jump_w"):
        with st.popover(f"No.{cow['No']} ✎"):
            nos = [c["No"] for c in st.session_state.cows]
            target_no = st.number_input(
                "出場番号を入力して移動", min_value=int(min(nos)), max_value=int(max(nos)),
                value=int(cow["No"]), step=1, key=f"jump_no_w_{idx}"
            )
            if st.button("この番号へ移動", key=f"jump_go_w_{idx}", use_container_width=True):
                target_idx = next((i for i, c in enumerate(st.session_state.cows) if c["No"] == target_no), None)
                if target_idx is not None:
                    if st.session_state.input_buffer:
                        st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                    st.session_state.curr_idx_w = target_idx
                    st.session_state.input_buffer = ""
                    st.rerun(scope="fragment")
                else:
                    st.warning("その出場番号は見つかりませんでした。")

    # 2. 上部：牛シルエット＆入力表示（モックアップの「体重入力画面」上段）
    display_w = st.session_state.input_buffer if st.session_state.input_buffer != "" else (str(cow["体重"]) if cow["体重"] > 0 else "")

    card_html_w = (
        '<div class="screen-card">'
        '<div class="card-top">'
        f'{get_cow_svg(cow["No"])}'
        '<div class="input-display-row">'
        f'<span class="input-display">{display_w}</span>'
        '<span class="input-unit">kg</span>'
        '</div>'
        '<div class="cow-meta">'
        f'性別: <b>{cow["性別"]}</b> ｜ 日齢: <b>{cow["日齢"]}日</b> ｜ 父: <b>{cow["父"]}</b>'
        '</div>'
        '</div>'
        '<div class="card-divider"></div>'
        '<div class="card-bottom" id="numpad-anchor-w"></div>'
        '</div>'
    )
    st.markdown(card_html_w, unsafe_allow_html=True)

    # 3. テンキー & 左右移動ボタン（カード下段を模したグリッド）
    with st.container(key="numpad_area_w"):
        col_l, col_pad, col_r = st.columns([0.8, 4.6, 0.8])

        with col_l:
            if st.button("←", key="prev_w", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                st.session_state.curr_idx_w = max(0, idx - 1)
                st.session_state.input_buffer = ""
                st.rerun(scope="fragment")

        with col_r:
            if st.button("→", key="next_w", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                st.session_state.curr_idx_w = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun(scope="fragment")

        with col_pad:
            for row_nums in [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]:
                cols = st.columns(3)
                for i, num in enumerate(row_nums):
                    if cols[i].button(num, key=f"btn_w_{num}", use_container_width=True):
                        st.session_state.input_buffer += num
                        st.rerun(scope="fragment")
            cols_bottom = st.columns(3)
            if cols_bottom[0].button("C", key="btn_w_c", use_container_width=True):
                st.session_state.input_buffer = ""
                st.session_state.cows[idx]["体重"] = 0
                st.rerun(scope="fragment")
            if cols_bottom[1].button("0", key="btn_w_0", use_container_width=True):
                st.session_state.input_buffer += "0"
                st.rerun(scope="fragment")
            if cols_bottom[2].button("決定", key="btn_w_enter", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                st.session_state.curr_idx_w = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun(scope="fragment")


with tab2:
    render_weight_tab()
# =========================================================
# 画面3: 落札価格入力画面（セリ本番）
# =========================================================
@st.fragment
def render_price_tab():
    total = len(st.session_state.cows)
    idx = st.session_state.curr_idx_p
    cow = st.session_state.cows[idx]
    calc = calculate_cow_metrics(cow)
    avg_profit = calculate_today_avg_profit()
    
    display_p = st.session_state.input_buffer if st.session_state.input_buffer != "" else (str(cow["実際落札額"]) if cow["実際落札額"] > 0 else "")

    # 1. No.Xをクリックすると番号を直接入力して任意の牛へ移動できる
    with st.container(key="no_jump_p"):
        with st.popover(f"No.{cow['No']} ✎"):
            nos = [c["No"] for c in st.session_state.cows]
            target_no = st.number_input(
                "出場番号を入力して移動", min_value=int(min(nos)), max_value=int(max(nos)),
                value=int(cow["No"]), step=1, key=f"jump_no_p_{idx}"
            )
            if st.button("この番号へ移動", key=f"jump_go_p_{idx}", use_container_width=True):
                target_idx = next((i for i, c in enumerate(st.session_state.cows) if c["No"] == target_no), None)
                if target_idx is not None:
                    if st.session_state.input_buffer:
                        st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                    st.session_state.curr_idx_p = target_idx
                    st.session_state.input_buffer = ""
                    st.rerun(scope="fragment")
                else:
                    st.warning("その出場番号は見つかりませんでした。")

    # 2. 上部：牛シルエット・日齢・体重・推定ボーダー・推定利益
    card_html_p = (
        '<div class="screen-card">'
        '<div class="card-top">'
        f'{get_cow_svg(cow["No"])}'
        '<div class="cow-meta">'
        f'日齢: <b>{cow["日齢"]}日</b><br>'
        f'体重: <b>{cow["体重"]}kg</b><br>'
        f'父: <b>{cow["父"]}</b><br>'
        f'摘要: <b>{cow.get("摘要", "") or "-"}</b>'
        '</div>'
        '<div class="cow-metrics">'
        f'本日の推定平均利益 <span class="profit">{avg_profit}</span>(千円)<br>'
        f'推定ボーダー価格 <span class="border-price">{calc["ボーダー価格"]}</span>(千円)'
        '</div>'
        '<div class="input-display-row">'
        f'<span class="input-display">{display_p}</span>'
        '<span class="input-unit">千円</span>'
        '</div>'
        '</div>'
        '<div class="card-divider"></div>'
        '</div>'
    )
    st.markdown(card_html_p, unsafe_allow_html=True)

    # 3. 購入チェック（テンキー真上の隙間に移動）
    with st.container(key="purchase_check_area_p"):
        purchased = st.checkbox("購入チェック", value=cow["自社落札"], key=f"buy_check_{idx}")
        st.session_state.cows[idx]["自社落札"] = purchased

    # 4. テンキー & 左右移動
    with st.container(key="numpad_area_p"):
        col_l, col_pad, col_r = st.columns([0.8, 4.6, 0.8])

        with col_l:
            if st.button("←", key="prev_p", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                st.session_state.curr_idx_p = max(0, idx - 1)
                st.session_state.input_buffer = ""
                st.rerun(scope="fragment")

        with col_r:
            if st.button("→", key="next_p", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                st.session_state.curr_idx_p = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun(scope="fragment")

        with col_pad:
            for row_nums in [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]:
                cols = st.columns(3)
                for i, num in enumerate(row_nums):
                    if cols[i].button(num, key=f"btn_p_{num}", use_container_width=True):
                        st.session_state.input_buffer += num
                        st.rerun(scope="fragment")
            cols_bottom = st.columns(3)
            if cols_bottom[0].button("C", key="btn_p_c", use_container_width=True):
                st.session_state.input_buffer = ""
                st.session_state.cows[idx]["実際落札額"] = 0
                st.rerun(scope="fragment")
            if cols_bottom[1].button("0", key="btn_p_0", use_container_width=True):
                st.session_state.input_buffer += "0"
                st.rerun(scope="fragment")
            if cols_bottom[2].button("決定", key="btn_p_enter", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                st.session_state.curr_idx_p = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun(scope="fragment")


with tab3:
    render_price_tab()
# =========================================================
# 画面4: セリ結果一覧表示画面
# =========================================================
with tab4:
    st.subheader("📋 セリ結果一覧表示画面")
    
    # 1. 本日落札した牛一覧
    my_cows = [c for c in st.session_state.cows if c.get("自社落札", False)]
    st.markdown("#### 🏆 本日落札した牛一覧")
    if my_cows:
        df_my = pd.DataFrame(my_cows)[[
            "No", "性別", "生年月日", "日齢", "産次", "体重",
            "父", "母の父", "母の祖父", "母の母の祖父", "摘要", "実際落札額"
        ]]
        df_my.columns = [
            "出場番号", "性別", "生年月日", "日齢", "産次", "当日体重(kg)",
            "父牛", "母の父", "母の祖父", "母の母の祖父", "摘要", "落札額(千円)"
        ]
        st.dataframe(df_my, use_container_width=True, hide_index=True)
    else:
        st.info("自社落札した牛はまだありません。")
        
    st.divider()
    
    # 2. 本日のセリ結果一覧（全頭）
    st.markdown("#### 📑 本日のセリ結果一覧（全頭）")
    all_rows = []
    for c in st.session_state.cows:
        m = calculate_cow_metrics(c)
        all_rows.append({
            "出場番号": c["No"],
            "性別": c["性別"],
            "生年月日": c.get("生年月日", "-"),
            "日齢": c["日齢"],
            "産次": c.get("産次", "-"),
            "体重(kg)": c["体重"],
            "父": c["父"],
            "母の父": c.get("母の父", "-"),
            "母の祖父": c.get("母の祖父", "-"),
            "母の母の祖父": c.get("母の母の祖父", "-"),
            "摘要": c.get("摘要", "") or "-",
            "ボーダー(千円)": m["ボーダー価格"],
            "落札額(千円)": c["実際落札額"],
            "購入結果": "自社落札" if c.get("自社落札", False) else ("他社落札" if c["実際落札額"] > 0 else "-")
        })
    df_all = pd.DataFrame(all_rows)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # 3. kintone送信ボタン
    if st.button("☁️ タップでkintoneに送る", type="primary", use_container_width=True):
        with st.spinner("kintoneにデータを送信中..."):
            success, msg = send_to_kintone(st.session_state.cows)
            if success:
                st.success(msg)
            else:
                st.error(msg)