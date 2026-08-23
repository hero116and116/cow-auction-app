import streamlit as st
import pandas as pd
import json
import os
import textwrap
import requests
from google import genai
from google.genai import types

st.set_page_config(page_title="牛セリ適正価格チェッカー", page_icon="🐄", layout="centered")

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
    }
    div[data-testid="stTabs"] { position: static !important; }

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

    /* 牛のシルエット */
    .cow-icon-container {
        position: relative;
        display: inline-block;
        width: 150px;
        height: 95px;
        margin: 4px 0;
    }
    .cow-svg {
        width: 100%;
        height: 100%;
        fill: #334155;
    }
    .cow-number-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -40%);
        color: #ffffff;
        font-size: 26px;
        font-weight: 900;
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

    /* テンキー全体を囲むコンテナ：カード下段として視覚的につなげる */
    .st-key-numpad_area_w, .st-key-numpad_area_p {
        border: 2px solid #1e293b;
        border-top: none;
        border-radius: 0 0 4px 4px;
        margin-top: -16px;
        padding: 14px 10px 16px 10px;
        background-color: #ffffff;
    }

    /* --- テンキー行のレイアウト ---
       .st-key-numpad_area_w / .st-key-numpad_area_p は
       st.container(key=...) が実際に生成するDOM上の親要素につく
       クラスなので、ここを起点にスマホ幅でも横並びを強制する。 */
    .st-key-numpad_area_w div[data-testid="stHorizontalBlock"],
    .st-key-numpad_area_p div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 4px !important;
    }
    .st-key-numpad_area_w div[data-testid="stColumn"],
    .st-key-numpad_area_p div[data-testid="stColumn"],
    .st-key-numpad_area_w div[data-testid="column"],
    .st-key-numpad_area_p div[data-testid="column"] {
        width: auto !important;
        min-width: 0 !important;
        flex: 1 1 0 !important;
    }
    .st-key-numpad_area_w button,
    .st-key-numpad_area_p button {
        height: 72px !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        padding: 0 !important;
    }
    .st-key-numpad_area_w button:hover,
    .st-key-numpad_area_p button:hover { border-color: #3b82f6 !important; color: #3b82f6 !important; }

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
</style>
""", unsafe_allow_html=True)

# --- APIキー & kintone設定 ---
GEMINI_API_KEY = "AQ.Ab8RN6KGaI97aQ0liR_8kYw5ALr-SMS8KzDW8cPaMUnlt4veDQ"
KINTONE_DOMAIN = "cattlook.cybozu.com"
KINTONE_APP_ID = "131"
KINTONE_API_TOKEN = "T4aTJyzRN736eaqzzWucxZIIbXy9wYn5YkAnlJsO"

# --- サイドバー設定 ---
with st.sidebar:
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
        return {"DG": 0.0, "育成日数": 0, "育成コスト": 0, "予測出荷体重": 0.0, "予測枝肉重量": 0.0, "見込売上": 0, "ボーダー価格": 0, "推定利益": 0}
    
    dg = (weight - birth_weight) / days
    raising_days = max(0, shipment_days - days)
    cost = int(raising_days * daily_cost)
    pred_ship_weight = weight + (dg * raising_days)
    pred_carcass_weight = pred_ship_weight * yield_rate
    sales = int(pred_carcass_weight * carcass_price)
    border_price = max(0, (sales - cost) // 1000)
    
    # 想定利益（売上の約15〜20%目安または目標粗利）
    estimated_profit = max(0, int(sales * 0.15) // 1000)
    
    return {
        "DG": round(dg, 3),
        "育成日数": int(raising_days),
        "育成コスト": cost // 1000,
        "予測出荷体重": round(pred_ship_weight, 1),
        "予測枝肉重量": round(pred_carcass_weight, 1),
        "見込売上": sales // 1000,
        "ボーダー価格": border_price,
        "推定利益": estimated_profit
    }

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
    キー: No (整数), 性別 (去/雌), 日齢 (整数), 産次 (整数、無ければ0), 摘要 (文字列), 父 (文字列), 母の父 (文字列), 母の祖父 (文字列), 母の母の祖父 (文字列)
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
        {"No": i, "体重": 0, "実際落札額": 0, "性別": "去", "日齢": 280, "産次": 1, "父": "福勝鶴", "母の父": "美津照重", "母の祖父": "平茂勝", "母の母の祖父": "-", "摘要": "", "自社落札": False}
        for i in range(1, 31)
    ]
if "curr_idx" not in st.session_state:
    st.session_state.curr_idx = 0
if "input_buffer" not in st.session_state:
    st.session_state.input_buffer = ""

# --- 牛のシルエットSVGヘルパー ---
def get_cow_svg(number_str):
    html = f"""<div class="cow-icon-container">
<svg class="cow-svg" viewBox="0 0 100 65">
<path d="M88,25 C88,20 80,15 75,18 C70,12 65,12 60,15 C55,14 35,14 25,20 C18,25 12,30 10,40 C10,48 15,55 20,55 C22,55 24,48 26,48 C28,48 30,55 35,55 C38,55 40,50 42,48 C45,48 55,48 60,52 C62,55 68,55 70,48 C72,48 76,55 80,55 C85,55 88,45 88,38 C92,35 95,28 92,24 C89,22 88,25 88,25 Z"/>
</svg>
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
                st.session_state.curr_idx = 0
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
    idx = st.session_state.curr_idx
    cow = st.session_state.cows[idx]
    
    # 1. 上部：牛シルエット＆No（モックアップの「体重入力画面」上段）
    display_w = st.session_state.input_buffer if st.session_state.input_buffer != "" else (str(cow["体重"]) if cow["体重"] > 0 else "")

    card_html_w = (
        '<div class="screen-card">'
        '<div class="card-top">'
        f'<div class="cow-no-label">No.{cow["No"]}</div>'
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

    # 2. テンキー & 左右移動ボタン（カード下段を模したグリッド）
    with st.container(key="numpad_area_w"):
        col_l, col_pad, col_r = st.columns([1.1, 3.8, 1.1])

        with col_l:
            if st.button("←", key="prev_w", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                st.session_state.curr_idx = max(0, idx - 1)
                st.session_state.input_buffer = ""
                st.rerun()

        with col_r:
            if st.button("→", key="next_w", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                st.session_state.curr_idx = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun()

        with col_pad:
            for row_nums in [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]:
                cols = st.columns(3)
                for i, num in enumerate(row_nums):
                    if cols[i].button(num, key=f"btn_w_{num}", use_container_width=True):
                        st.session_state.input_buffer += num
                        st.rerun()
            cols_bottom = st.columns(3)
            if cols_bottom[0].button("C", key="btn_w_c", use_container_width=True):
                st.session_state.input_buffer = ""
                st.session_state.cows[idx]["体重"] = 0
                st.rerun()
            if cols_bottom[1].button("0", key="btn_w_0", use_container_width=True):
                st.session_state.input_buffer += "0"
                st.rerun()
            if cols_bottom[2].button("決定", key="btn_w_enter", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
                st.session_state.curr_idx = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun()


with tab2:
    render_weight_tab()
# =========================================================
# 画面3: 落札価格入力画面（セリ本番）
# =========================================================
@st.fragment
def render_price_tab():
    total = len(st.session_state.cows)
    idx = st.session_state.curr_idx
    cow = st.session_state.cows[idx]
    calc = calculate_cow_metrics(cow)
    
    display_p = st.session_state.input_buffer if st.session_state.input_buffer != "" else (str(cow["実際落札額"]) if cow["実際落札額"] > 0 else "")

    # 1. 購入チェック（モックアップ右上の「購入チェック」枠）
    col_title, col_check = st.columns([4, 1])
    with col_title:
        st.markdown(f"<div class='cow-no-label' style='margin-top:6px;'>No.{cow['No']}</div>", unsafe_allow_html=True)
    with col_check:
        purchased = st.checkbox("購入チェック", value=cow["自社落札"], key=f"buy_check_{idx}")
        st.session_state.cows[idx]["自社落札"] = purchased

    # 2. 上部：牛シルエット・日齢・体重・推定ボーダー・推定利益
    card_html_p = (
        '<div class="screen-card">'
        '<div class="card-top">'
        f'{get_cow_svg(cow["No"])}'
        '<div class="cow-meta">'
        f'日齢: <b>{cow["日齢"]}日</b><br>'
        f'体重: <b>{cow["体重"]}kg</b><br>'
        f'父: <b>{cow["父"]}</b>'
        '</div>'
        '<div class="cow-metrics">'
        f'本日の推定平均利益　<span class="profit">{calc["推定利益"]}</span>(千円)<br>'
        f'推定ボーダー価格　<span class="border-price">{calc["ボーダー価格"]}</span>(千円)'
        '</div>'
        '<div class="input-display-row">'
        f'<span class="input-display">{display_p}</span>'
        '<span class="input-unit">千円</span>'
        '</div>'
        '</div>'
        '<div class="card-divider"></div>'
        '<div class="card-bottom"></div>'
        '</div>'
    )
    st.markdown(card_html_p, unsafe_allow_html=True)

    # 3. テンキー & 左右移動
    with st.container(key="numpad_area_p"):
        col_l, col_pad, col_r = st.columns([1.1, 3.8, 1.1])

        with col_l:
            if st.button("←", key="prev_p", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                st.session_state.curr_idx = max(0, idx - 1)
                st.session_state.input_buffer = ""
                st.rerun()

        with col_r:
            if st.button("→", key="next_p", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                st.session_state.curr_idx = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun()

        with col_pad:
            for row_nums in [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]:
                cols = st.columns(3)
                for i, num in enumerate(row_nums):
                    if cols[i].button(num, key=f"btn_p_{num}", use_container_width=True):
                        st.session_state.input_buffer += num
                        st.rerun()
            cols_bottom = st.columns(3)
            if cols_bottom[0].button("C", key="btn_p_c", use_container_width=True):
                st.session_state.input_buffer = ""
                st.session_state.cows[idx]["実際落札額"] = 0
                st.rerun()
            if cols_bottom[1].button("0", key="btn_p_0", use_container_width=True):
                st.session_state.input_buffer += "0"
                st.rerun()
            if cols_bottom[2].button("決定", key="btn_p_enter", use_container_width=True):
                if st.session_state.input_buffer:
                    st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
                st.session_state.curr_idx = min(total - 1, idx + 1)
                st.session_state.input_buffer = ""
                st.rerun()


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
        df_my = pd.DataFrame(my_cows)[["No", "性別", "日齢", "体重", "父", "実際落札額"]]
        df_my.columns = ["出場番号", "性別", "日齢", "当日体重(kg)", "父牛", "落札額(千円)"]
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
            "日齢": c["日齢"],
            "性別": c["性別"],
            "体重(kg)": c["体重"],
            "父": c["父"],
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