import streamlit as st
import pandas as pd
import json
from google import genai
from google.genai import types

st.set_page_config(page_title="牛セリ適正価格チェッカー", layout="wide")
st.title("🐄 牛セリ落札上限価格シミュレーター")

# --- APIキーの設定 ---
GEMINI_API_KEY = "AQ.Ab8RN6KGaI97aQ0liR_8kYw5ALr-SMS8KzDW8cPaMUnlt4veDQ"

# --- サイドバー：共通設定パラメータ ---
st.sidebar.header("⚙️ 共通設定（相場・コスト）")
carcass_price = st.sidebar.number_input("枝肉単価 (円/kg)", value=2500, step=50)
daily_cost = st.sidebar.number_input("1日あたり育成コスト (円)", value=850, step=10)
shipment_days = st.sidebar.number_input("出荷日齢 (日)", value=854, step=1)
birth_weight = st.sidebar.number_input("生時体重 (kg)", value=35.0, step=1.0)
yield_rate = st.sidebar.number_input("歩留基準 (0.65 = 65%)", value=0.65, step=0.01)

# --- 計算ロジック関数 ---
def calculate_cow(row):
    try:
        days = float(row.get("日齢", 0))
        weight = float(row.get("体重", 0))
    except (ValueError, TypeError):
        days, weight = 0, 0
    
    if days <= 0 or weight <= birth_weight:
        return pd.Series([0.0, 0, 0, 0.0, 0.0, 0, 0], 
                         index=["DG", "育成日数", "育成コスト(千円)", "予測出荷体重", "予測枝肉重量", "見込売上(千円)", "上限価格(千円)"])
    
    # 1. DG
    dg = (weight - birth_weight) / days
    # 2. 育成日数
    raising_days = max(0, shipment_days - days)
    # 3. 育成コスト
    cost = int(raising_days * daily_cost)
    # 4. 予測出荷体重
    increased_weight = dg * raising_days
    pred_ship_weight = weight + increased_weight
    # 5. 予測枝肉重量
    pred_carcass_weight = pred_ship_weight * yield_rate
    # 6. 売上 & 上限価格
    sales = int(pred_carcass_weight * carcass_price)
    limit_price = sales - cost
    
    return pd.Series([
        round(dg, 3), 
        int(raising_days), 
        cost // 1000, 
        round(pred_ship_weight, 1), 
        round(pred_carcass_weight, 1), 
        sales // 1000, 
        limit_price // 1000
    ], index=["DG", "育成日数", "育成コスト(千円)", "予測出荷体重", "予測枝肉重量", "見込売上(千円)", "上限価格(千円)"])

# --- AI画像/PDF解析関数（PILを使わずバイナリ直接送信） ---
def parse_catalog_file(uploaded_file, key=GEMINI_API_KEY):
    client = genai.Client(api_key=key)
    file_bytes = uploaded_file.getvalue()
    
    # ファイル形式の判定
    name_lower = uploaded_file.name.lower()
    if name_lower.endswith(".pdf"):
        mime_type = "application/pdf"
    elif name_lower.endswith(".png"):
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    prompt = """
    添付された牛のセリ名簿のデータから、各行の情報を抽出し、以下のキーを持つJSON配列として出力してください。
    - No: 出場番号 (整数)
    - 性別: 性別 (去 / 雌)
    - 日齢: 日齢 (整数)
    - 父: 血統の「父」の名前 (文字列)

    ※ 体重は当日に記入されるため 0 としてください。
    ※ 実際落札額(千円)は 0 としてください。
    ※ 余計な文章は一切出力せず、JSON配列のみを返してください。
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)

# --- セッションステート初期化 ---
if "initial_cows" not in st.session_state:
    st.session_state.initial_cows = pd.DataFrame([
        {"No": i, "性別": "去", "日齢": 280, "父": "-", "体重": 0.0, "実際落札額(千円)": 0}
        for i in range(1, 31)
    ])

# --- 名簿アップロード機能 ---
st.subheader("📷 名簿ファイル（写真 / PDF）から自動読み取り")
uploaded_files = st.file_uploader(
    "名簿ファイルを選択（複数可）", 
    type=["png", "jpg", "jpeg", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 AIで名簿を解析して表に展開"):
        all_parsed_rows = []
        with st.spinner("AIが名簿を解析中..."):
            for f in uploaded_files:
                parsed_data = parse_catalog_file(f)
                all_parsed_rows.extend(parsed_data)
            
            if all_parsed_rows:
                df_new = pd.DataFrame(all_parsed_rows)
                if "体重" not in df_new.columns:
                    df_new["体重"] = 0.0
                if "実際落札額(千円)" not in df_new.columns:
                    df_new["実際落札額(千円)"] = 0
                
                st.session_state.initial_cows = df_new
                st.success(f"合計 {len(df_new)} 頭分の名簿データを読み込みました！")
                st.rerun()

st.divider()

# --- データ入力エディタ ---
st.subheader("📝 牛データ入力（体重・落札額）")
st.caption("※ 体重を入力してEnterを押すと、即座に下に上限落札価格が算出されます。")

edited_df = st.data_editor(
    st.session_state.initial_cows,
    num_rows="dynamic",
    use_container_width=True,
    height=320,
    column_config={
        "No": st.column_config.NumberColumn("No", format="%d"),
        "性別": st.column_config.TextColumn("性別", width="small"),
        "日齢": st.column_config.NumberColumn("日齢 (日)", format="%d"),
        "父": st.column_config.TextColumn("父(血統)"),
        "体重": st.column_config.NumberColumn("当日体重 (kg)", format="%.1f"),
        "実際落札額(千円)": st.column_config.NumberColumn("実セリ値 (千円)", format="%d"),
    }
)

# --- 計算実行と表示 ---
calc_results = edited_df.apply(calculate_cow, axis=1)
result_df = pd.concat([edited_df, calc_results], axis=1)

def judge(row):
    actual = row.get("実際落札額(千円)", 0)
    limit = row.get("上限価格(千円)", 0)
    if actual <= 0 or row.get("体重", 0) <= 0:
        return "-"
    diff = limit - actual
    return f"🟢 ＋{diff:,}千円 (得)" if diff >= 0 else f"🔴 {diff:,}千円 (損)"

result_df["判定(差額)"] = result_df.apply(judge, axis=1)

# 結果テーブル
st.subheader("📊 判定・計算結果一覧")
active_cows = result_df[(result_df["体重"] > 0) | (result_df["No"] <= 5)]

st.dataframe(
    active_cows[[
        "No", "性別", "日齢", "父", "体重", "DG", 
        "上限価格(千円)", "実際落札額(千円)", "判定(差額)",
        "見込売上(千円)", "育成コスト(千円)", "予測出荷体重"
    ]],
    use_container_width=True,
    height=350
)
