import streamlit as st
import pandas as pd
import json
import os
import requests
from google import genai
from google.genai import types

st.set_page_config(page_title="牛セリ適正価格チェッカー", page_icon="🐄", layout="centered")

# --- カスタムCSS（専用テンキー・牛カードデザイン） ---
st.markdown("""
<style>
    /* 全体フォント・余白調整 */
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    
    /* 牛のシルエットカード */
    .cow-card {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        margin-bottom: 12px;
        position: relative;
    }
    .cow-icon-container {
        position: relative;
        display: inline-block;
        width: 140px;
        height: 90px;
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
    .input-display {
        font-size: 32px;
        font-weight: 800;
        color: #1e293b;
        border-bottom: 3px solid #3b82f6;
        display: inline-block;
        min-width: 140px;
        padding: 2px 10px;
        margin-top: 6px;
    }
    .status-badge {
        font-size: 15px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 4px;
    }
    
    /* テンキー・ナビゲーションボタン */
    div[data-testid="stHorizontalBlock"] button {
        height: 56px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
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
    return f"""
    <div class="cow-icon-container">
        <svg class="cow-svg" viewBox="0 0 100 65">
            <path d="M88,25 C88,20 80,15 75,18 C70,12 65,12 60,15 C55,14 35,14 25,20 C18,25 12,30 10,40 C10,48 15,55 20,55 C22,55 24,48 26,48 C28,48 30,55 35,55 C38,55 40,50 42,48 C45,48 55,48 60,52 C62,55 68,55 70,48 C72,48 76,55 80,55 C85,55 88,45 88,38 C92,35 95,28 92,24 C89,22 88,25 88,25 Z"/>
        </svg>
        <div class="cow-number-overlay">{number_str}</div>
    </div>
    """

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
                st.success(f"✅ {len(parsed)} 頭の名簿データを読み込みました！")
                st.rerun()

# =========================================================
# 画面2: 体重入力画面（下見）
# =========================================================
with tab2:
    total = len(st.session_state.cows)
    idx = st.session_state.curr_idx
    cow = st.session_state.cows[idx]
    
    # 1. 上部：牛シルエット＆No
    display_w = st.session_state.input_buffer if st.session_state.input_buffer != "" else (str(cow["体重"]) if cow["体重"] > 0 else "___")
    
    st.markdown(f"""
    <div class="cow-card">
        <h3 style="margin:0; color:#64748b;">No. {cow['No']}</h3>
        {get_cow_svg(cow['No'])}
        <div><span class="input-display">{display_w}</span> <span style="font-size:20px; font-weight:700;">kg</span></div>
        <div style="margin-top:8px; color:#475569; font-size:14px;">
            性別: <b>{cow['性別']}</b> ｜ 日齢: <b>{cow['日齢']}日</b> ｜ 父: <b>{cow['父']}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. テンキー & 左右移動ボタン
    col_l, col_pad, col_r = st.columns([1.2, 3.6, 1.2])
    
    with col_l:
        st.write("")
        st.write("")
        if st.button("⬅️", key="prev_w", use_container_width=True):
            if st.session_state.input_buffer:
                st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
            st.session_state.curr_idx = max(0, idx - 1)
            st.session_state.input_buffer = ""
            st.rerun()
            
    with col_r:
        st.write("")
        st.write("")
        if st.button("➡️", key="next_w", use_container_width=True):
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
        if cols_bottom[2].button("💾 決定", key="btn_w_enter", use_container_width=True, type="primary"):
            if st.session_state.input_buffer:
                st.session_state.cows[idx]["体重"] = float(st.session_state.input_buffer)
            st.session_state.curr_idx = min(total - 1, idx + 1)
            st.session_state.input_buffer = ""
            st.rerun()

# =========================================================
# 画面3: 落札価格入力画面（セリ本番）
# =========================================================
with tab3:
    total = len(st.session_state.cows)
    idx = st.session_state.curr_idx
    cow = st.session_state.cows[idx]
    calc = calculate_cow_metrics(cow)
    
    display_p = st.session_state.input_buffer if st.session_state.input_buffer != "" else (str(cow["実際落札額"]) if cow["実際落札額"] > 0 else "___")
    
    # 1. 上部：牛シルエット・日齢・体重・推定ボーダー・推定利益
    st.markdown(f"""
    <div class="cow-card">
        <h3 style="margin:0; color:#64748b;">No. {cow['No']}</h3>
        <div style="display:flex; justify-content:space-around; align-items:center;">
            <div style="text-align:left; font-size:15px; font-weight:700; color:#334155;">
                日齢: {cow['日齢']}日<br>
                体重: {cow['体重']}kg<br>
                父: {cow['父']}
            </div>
            {get_cow_svg(cow['No'])}
        </div>
        <div style="margin-top:8px; font-size:16px; font-weight:700; color:#0f172a;">
            本日の推定平均利益: <span style="color:#059669; font-size:18px;">{calc['推定利益']}</span> (千円)<br>
            推定ボーダー価格: <span style="color:#2563eb; font-size:20px;">{calc['ボーダー価格']}</span> (千円)
        </div>
        <div><span class="input-display">{display_p}</span> <span style="font-size:20px; font-weight:700;">千円</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. 購入チェックボックス（右上に配置）
    purchased = st.checkbox("⭐ 自社で購入（購入チェック）", value=cow["自社落札"], key=f"buy_check_{idx}")
    st.session_state.cows[idx]["自社落札"] = purchased
    
    # 3. テンキー & 左右移動
    col_l, col_pad, col_r = st.columns([1.2, 3.6, 1.2])
    
    with col_l:
        st.write("")
        st.write("")
        if st.button("⬅️", key="prev_p", use_container_width=True):
            if st.session_state.input_buffer:
                st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
            st.session_state.curr_idx = max(0, idx - 1)
            st.session_state.input_buffer = ""
            st.rerun()
            
    with col_r:
        st.write("")
        st.write("")
        if st.button("➡️", key="next_p", use_container_width=True):
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
        if cols_bottom[2].button("💾 決定", key="btn_p_enter", use_container_width=True, type="primary"):
            if st.session_state.input_buffer:
                st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer)
            st.session_state.curr_idx = min(total - 1, idx + 1)
            st.session_state.input_buffer = ""
            st.rerun()

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