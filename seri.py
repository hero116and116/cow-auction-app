import streamlit as st
import pandas as pd
import json
import os
from google import genai
from google.genai import types

st.set_page_config(page_title="牛セリ適正価格チェッカー", layout="wide")
st.title("🐄 牛セリ落札上限価格シミュレーター")

# --- APIキーの設定 ---
GEMINI_API_KEY = "AQ.Ab8RN6KGaI97aQ0liR_8kYw5ALr-SMS8KzDW8cPaMUnlt4veDQ"

# --- サイドバー：共通設定パラメータ ---
with st.sidebar:
    st.header("⚙️ 共通設定（相場・コスト）")
    carcass_price = st.number_input("枝肉単価 (円/kg)", value=2500, step=50)
    daily_cost = st.number_input("1日あたり育成コスト (円)", value=850, step=10)
    shipment_days = st.number_input("出荷日齢 (日)", value=854, step=1)
    birth_weight = st.number_input("生時体重 (kg)", value=35.0, step=1.0)
    yield_rate = st.number_input("歩留基準 (0.65 = 65%)", value=0.65, step=0.01)

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
    
    dg = (weight - birth_weight) / days
    raising_days = max(0, shipment_days - days)
    cost = int(raising_days * daily_cost)
    increased_weight = dg * raising_days
    pred_ship_weight = weight + increased_weight
    pred_carcass_weight = pred_ship_weight * yield_rate
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

# --- AI画像/PDF解析関数 ---
def parse_catalog_file(uploaded_file, key=GEMINI_API_KEY):
    client = genai.Client(api_key=key)
    file_bytes = uploaded_file.getvalue()
    
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
if "cows_df" not in st.session_state:
    st.session_state.cows_df = pd.DataFrame([
        {"No": i, "体重": 0.0, "性別": "去", "日齢": 280, "父": "-", "実際落札額(千円)": 0}
        for i in range(1, 31)
    ])

if "current_no_idx" not in st.session_state:
    st.session_state.current_no_idx = 0

# --- 1. 名簿アップロード ---
with st.expander("📷 名簿ファイル（写真 / PDF）の自動読み取り", expanded=False):
    uploaded_files = st.file_uploader(
        "名簿ファイルを選択（複数可）", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True
    )
    if uploaded_files and st.button("🚀 AIで名簿を解析して表に展開"):
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
                
                # 並び順を「No」「体重」を先頭に整理
                df_new = df_new[["No", "体重", "性別", "日齢", "父", "実際落札額(千円)"]]
                st.session_state.cows_df = df_new
                st.session_state.current_no_idx = 0
                st.success(f"合計 {len(df_new)} 頭分の名簿データを登録しました！")
                st.rerun()

st.divider()

# --- 2. スマホ専用：下見 連続体重入力カード ---
st.subheader("📱 下見モード（1頭ずつ連続入力）")

df = st.session_state.cows_df
total_cows = len(df)

if total_cows > 0:
    # 選択中の牛のインデックス制御
    idx = st.session_state.current_no_idx
    if idx >= total_cows:
        idx = total_cows - 1
        st.session_state.current_no_idx = idx

    current_cow = df.iloc[idx]
    
    # 対象牛の情報を大きく表示
    col_nav1, col_nav2 = st.columns([1, 1])
    with col_nav1:
        st.markdown(f"### 🐂 **No. {int(current_cow['No'])}** ({idx + 1}/{total_cows}頭目)")
    with col_nav2:
        st.markdown(f"**性別**: {current_cow['性別']} ｜ **日齢**: {int(current_cow['日齢'])}日 ｜ **父**: {current_cow['父']}")

    # 入力フォーム
    with st.form(key=f"quick_input_form_{idx}"):
        input_w = st.number_input(
            "当日体重 (kg)", 
            value=float(current_cow["体重"]) if float(current_cow["体重"]) > 0 else 280.0, 
            step=1.0,
            format="%.1f"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_next = st.form_submit_button("💾 登録して次の牛へ ⏩", use_container_width=True, type="primary")
        with col_btn2:
            prev_btn = st.form_submit_button("⬅️ 前の牛に戻る", use_container_width=True)

        if submit_next:
            st.session_state.cows_df.at[idx, "体重"] = input_w
            if idx + 1 < total_cows:
                st.session_state.current_no_idx = idx + 1
            st.rerun()
            
        if prev_btn:
            if idx > 0:
                st.session_state.current_no_idx = idx - 1
            st.rerun()

st.divider()

# --- 3. 一覧表エディタ & 判定結果 ---
st.subheader("📊 セリ一覧表 & 上限価格判定")

# No と 体重 を先頭に配置
edited_df = st.data_editor(
    st.session_state.cows_df,
    num_rows="dynamic",
    use_container_width=True,
    height=280,
    column_config={
        "No": st.column_config.NumberColumn("No", format="%d", width="small"),
        "体重": st.column_config.NumberColumn("当日体重(kg)", format="%.1f", width="medium"),
        "性別": st.column_config.TextColumn("性別", width="small"),
        "日齢": st.column_config.NumberColumn("日齢", format="%d", width="small"),
        "父": st.column_config.TextColumn("父", width="medium"),
        "実際落札額(千円)": st.column_config.NumberColumn("実落札(千円)", format="%d", width="medium"),
    }
)
st.session_state.cows_df = edited_df

# 計算ロジック適用
calc_results = edited_df.apply(calculate_cow, axis=1)
result_df = pd.concat([edited_df, calc_results], axis=1)

def judge(row):
    actual = row.get("実際落札額(千円)", 0)
    limit = row.get("上限価格(千円)", 0)
    if actual <= 0 or row.get("体重", 0) <= 0:
        return "-"
    diff = limit - actual
    return f"🟢 ＋{diff:,}千円" if diff >= 0 else f"🔴 {diff:,}千円"

result_df["判定"] = result_df.apply(judge, axis=1)

# セリ本番用ビュー（体重が入力されている牛を上位表示）
st.dataframe(
    result_df[[
        "No", "上限価格(千円)", "判定", "体重", "DG", 
        "実際落札額(千円)", "見込売上(千円)", "育成コスト(千円)", "父", "日齢"
    ]],
    use_container_width=True,
    height=350
)