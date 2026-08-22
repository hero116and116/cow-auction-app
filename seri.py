import json
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

st.set_page_config(
    page_title="牛セリ適正価格チェッカー", page_icon="🐄", layout="wide"
)
st.title("🐄 牛セリ落札上限価格シミュレーター（統計分析強化版）")

# --- APIキーの設定 ---
GEMINI_API_KEY = "AQ.Ab8RN6KGaI97aQ0liR_8kYw5ALr-SMS8KzDW8cPaMUnlt4veDQ"

# --- kintone接続設定 ---
KINTONE_DOMAIN = "cattlook.cybozu.com"
KINTONE_APP_ID = "131"
KINTONE_API_TOKEN = "T4aTJyzRN736eaqzzWucxZIIbXy9wYn5YkAnlJsO"

# --- 1,121頭の実証分析から得られた統計回帰パラメータ ---
STAT_INTERCEPT = 470.64  # ベースライン（去勢・F1・その他血統）

# 父牛の枝肉重量補正値 (kg)
FATHER_EFFECTS = {
    "勝美糸": 28.21,
    "隆之姫": 18.70,
    "梅栄福": 10.27,
    "福雲勝": -9.02,
    "菊美津照": -15.68,
    "美照福": -28.45,
    "種牛": -29.28,
    "隆安幸": -37.36,
}

# 畜種の枝肉重量補正値 (kg) - 基準はF1
BREED_EFFECTS = {
    "F1": 0.0,
    "黒毛": 7.75,
    "F1クロス": -21.81,
    "褐毛": -26.85,
    "JF": -77.24,
    "H": 0.0,
}

GENDER_FEMALE_EFFECT = -37.14  # 雌牛のマイナス補正 (kg)

# --- サイドバー：共通設定パラメータ ---
with st.sidebar:
    st.header("⚙️ 共通設定（相場・コスト・目標利益）")
    carcass_price = st.number_input(
        "枝肉単価 (円/kg)", value=2500, step=50
    )
    daily_cost = st.number_input(
        "1日あたり育成コスト (円)", value=850, step=10
    )
    shipment_days = st.number_input("目標出荷日齢 (日)", value=854, step=1)
    birth_weight = st.number_input("生時推定体重 (kg)", value=35.0, step=1.0)
    target_profit = st.number_input(
        "目標確保利益 (千円/頭)",
        value=50,
        step=10,
        help="上限価格から差し引く利益マージン",
    )

    st.markdown("---")
    st.subheader("🧬 統計モデル基準値")
    st.caption(
        f"基準枝肉重量: **{STAT_INTERCEPT:.1f} kg** (去勢・F1・その他血統)"
    )
    st.caption(f"性別補正（雌）: **{GENDER_FEMALE_EFFECT:.1f} kg**")


# --- 性別・畜種正規化関数 ---
def clean_gender(val):
    s = str(val).strip()
    if "雌" in s or "メス" in s or "めす" in s or "女" in s:
        return "雌"
    return "去"


def clean_breed(val):
    s = str(val).strip()
    for b in BREED_EFFECTS.keys():
        if b in s:
            return b
    return "F1"


# --- 計算ロジック関数（統計回帰 ＋ 個体発育補正） ---
def calculate_cow(row):
    try:
        days = float(row.get("日齢", 0))
        weight = float(row.get("体重", 0))
    except (ValueError, TypeError):
        days, weight = 0, 0

    gender = clean_gender(row.get("性別", "去"))
    father = str(row.get("父", "")).strip()
    breed = clean_breed(row.get("畜種", "F1"))

    if days <= 0:
        return pd.Series(
            [0.0, 0, 0, 0.0, 0, 0],
            index=[
                "DG",
                "残育成日数",
                "育成コスト(千円)",
                "予測枝肉重量",
                "見込売上(千円)",
                "上限価格(千円)",
            ],
        )

    # 1. 導入時DGの算出
    if weight > birth_weight:
        dg = (weight - birth_weight) / days
    else:
        dg = 0.80  # 体重未入力時のデフォルトDG

    # 2. 統計回帰モデルによるベース予測（血統・性別・畜種）
    pred_carcass = STAT_INTERCEPT

    # 父牛補正
    father_matched = False
    for f_name, f_val in FATHER_EFFECTS.items():
        if f_name in father:
            pred_carcass += f_val
            father_matched = True
            break

    # 性別補正
    if gender == "雌":
        pred_carcass += GENDER_FEMALE_EFFECT

    # 畜種補正
    pred_carcass += BREED_EFFECTS.get(breed, 0.0)

    # 3. 個体発育（DG）による微補正（基準DG: 0.85kg/日からの乖離を反映）
    if weight > birth_weight:
        dg_diff = dg - 0.85
        pred_carcass += dg_diff * 40.0  # DGが0.1上がると枝肉重量約4kg増

    pred_carcass = max(200.0, round(pred_carcass, 1))

    # 4. コスト・売上・落札上限価格の算出
    raising_days = max(0, shipment_days - days)
    raising_cost = int(raising_days * daily_cost)
    sales = int(pred_carcass * carcass_price)

    # 上限価格 = 見込売上 - 残り育成コスト - 目標利益
    limit_price = sales - raising_cost - (target_profit * 1000)
    limit_price = max(0, limit_price)

    return pd.Series(
        [
            round(dg, 3),
            int(raising_days),
            raising_cost // 1000,
            pred_carcass,
            sales // 1000,
            limit_price // 1000,
        ],
        index=[
            "DG",
            "残育成日数",
            "育成コスト(千円)",
            "予測枝肉重量",
            "見込売上(千円)",
            "上限価格(千円)",
        ],
    )


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
    添付された牛のセリ名簿データから各牛の情報を抽出し、以下のキーを持つJSON配列として出力してください。
    【重要：各列の抽出ルール】
    - No: 出場番号 (整数)
    - 畜種: 「畜種」欄の値 (F1, 黒毛, 褐毛, JF, H など)
    - 性別: 「性別」欄の値。「去」または「雌」のどちらか1文字を正確に判定してください。
    - 日齢: 日齢 (整数)
    - 産次: 産次 (整数、記載がなければ 0)
    - 摘要: 摘要 (文字列、記載がなければ "")
    - 父: 血統の「父」の名前 (文字列)
    - 母の父: 血統の「母の父母」または「母の父」の名前 (文字列)
    - 母の祖父: 血統の「の祖父母」または「母の祖父」の名前 (文字列)
    - 母の母の祖父: 血統の「の母の祖父」の名前 (文字列)

    ※ 体重は 0 としてください。
    ※ 実際落札額(千円)は 0 としてください。
    ※ 余計な文章は一切出力せず、JSON配列のみを返してください。
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    data = json.loads(response.text)
    for row in data:
        row["性別"] = clean_gender(row.get("性別", "去"))
        row["畜種"] = clean_breed(row.get("畜種", "F1"))
    return data


# --- kintone一括保存関数 ---
def send_to_kintone(df):
    url = f"https://{KINTONE_DOMAIN}/k/v1/records.json"
    headers = {
        "X-Cybozu-API-Token": KINTONE_API_TOKEN,
        "Content-Type": "application/json",
    }

    records = []
    for _, row in df.iterrows():
        weight = float(row.get("体重", 0))
        actual_price = int(row.get("実際落札額(千円)", 0))
        is_purchased = bool(row.get("自社落札", False))

        if weight > 0 or actual_price > 0:
            status = "購入" if is_purchased else "非購入"
            gender = clean_gender(row.get("性別", "去"))

            records.append({
                "出場番号": {"value": int(row["No"])},
                "当日体重": {"value": weight},
                "日齢": {"value": int(row.get("日齢", 0))},
                "産次": {"value": int(row.get("産次", 0))},
                "摘要": {"value": str(row.get("摘要", ""))},
                "性別": {"value": gender},
                "父": {"value": str(row.get("父", ""))},
                "母の父": {"value": str(row.get("母の父", ""))},
                "母の祖父": {"value": str(row.get("母の祖父", ""))},
                "母の母の祖父": {"value": str(row.get("母の母の祖父", ""))},
                "落札上限価格": {"value": int(row.get("上限価格(千円)", 0))},
                "実際落札額": {"value": actual_price},
                "購入結果": {"value": status},
            })

    if not records:
        return (
            False,
            "保存対象のデータがありません（体重または実際落札額を入力してください）。",
        )

    payload = {"app": KINTONE_APP_ID, "records": records}

    try:
        res = requests.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            return (
                True,
                f"✅ {len(records)} 頭のセリ結果をkintoneに保存しました！",
            )
        else:
            return False, f"❌ 送信エラー ({res.status_code}): {res.text}"
    except Exception as e:
        return False, f"❌ 通信エラー: {str(e)}"


# --- セッションステート初期化 ---
if "cows_df" not in st.session_state:
    st.session_state.cows_df = pd.DataFrame([
        {
            "No": i,
            "体重": 0.0,
            "畜種": "F1",
            "性別": "去",
            "日齢": 280,
            "産次": 0,
            "摘要": "",
            "父": "-",
            "母の父": "-",
            "母の祖父": "-",
            "母の母の祖父": "-",
            "実際落札額(千円)": 0,
            "自社落札": False,
        }
        for i in range(1, 31)
    ])

if "current_no_idx" not in st.session_state:
    st.session_state.current_no_idx = 0

# --- 1. 名簿アップロード ---
with st.expander("📷 名簿ファイル（写真 / PDF）の自動読み取り", expanded=False):
    uploaded_files = st.file_uploader(
        "名簿ファイルを選択（複数可）",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("🚀 AIで名簿を解析して表に展開"):
        all_parsed_rows = []
        with st.spinner("AIが名簿を詳細解析中..."):
            for f in uploaded_files:
                parsed_data = parse_catalog_file(f)
                all_parsed_rows.extend(parsed_data)

            if all_parsed_rows:
                df_new = pd.DataFrame(all_parsed_rows)
                for col, default_val in [
                    ("体重", 0.0),
                    ("畜種", "F1"),
                    ("実際落札額(千円)", 0),
                    ("自社落札", False),
                    ("産次", 0),
                    ("摘要", ""),
                    ("父", "-"),
                    ("母の父", "-"),
                    ("母の祖父", "-"),
                    ("母の母の祖父", "-"),
                ]:
                    if col not in df_new.columns:
                        df_new[col] = default_val

                cols_order = [
                    "No",
                    "体重",
                    "畜種",
                    "性別",
                    "日齢",
                    "産次",
                    "父",
                    "母の父",
                    "母の祖父",
                    "母の母の祖父",
                    "摘要",
                    "実際落札額(千円)",
                    "自社落札",
                ]
                st.session_state.cows_df = df_new[cols_order]
                st.session_state.current_no_idx = 0
                st.success(
                    f"合計 {len(df_new)} 頭分の詳細データを展開しました！"
                )
                st.rerun()

st.divider()

# --- 2. スマホ専用：連続入力カード ---
st.subheader("📱 スマホ入力モード（下見・落札入力）")

df = st.session_state.cows_df
total_cows = len(df)

if total_cows > 0:
    idx = st.session_state.current_no_idx
    if idx >= total_cows:
        idx = total_cows - 1
        st.session_state.current_no_idx = idx

    current_cow = df.iloc[idx]
    current_w = float(current_cow["体重"])
    current_price = int(current_cow.get("実際落札額(千円)", 0))
    current_purchased = bool(current_cow.get("自社落札", False))

    temp_calc = calculate_cow(current_cow)
    limit_val = temp_calc["上限価格(千円)"]
    pred_cw = temp_calc["予測枝肉重量"]

    col_nav1, col_nav2 = st.columns([1, 1])
    with col_nav1:
        st.markdown(
            f"### 🐂 **No. {int(current_cow['No'])}** ({idx + 1}/{total_cows}頭目)"
        )
        if limit_val > 0:
            st.markdown(
                f"🎯 **落札上限目安: {limit_val:,} 千円** (予測枝肉: {pred_cw}kg)"
            )
    with col_nav2:
        st.markdown(
            f"**畜種**: `{current_cow.get('畜種', 'F1')}` ｜ **性別**: `{current_cow['性別']}` ｜ **日齢**: {int(current_cow['日齢'])}日\n\n"
            f"**父**: **{current_cow['父']}** ｜ **母父**: {current_cow['母の父']}"
        )

    with st.form(key=f"quick_input_form_{idx}"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            input_w = st.number_input(
                "当日体重 (kg)",
                value=current_w if current_w > 0 else None,
                step=1.0,
                format="%.1f",
                placeholder="体重 (例: 295)",
                key=f"weight_input_{idx}",
            )
        with col_in2:
            input_price = st.number_input(
                "実際落札額 (千円)",
                value=current_price if current_price > 0 else None,
                step=1,
                format="%d",
                placeholder="落札額 (例: 650)",
                key=f"price_input_{idx}",
            )

        input_purchased = st.checkbox(
            "⭐ 自社で落札した（購入）",
            value=current_purchased,
            key=f"purchased_check_{idx}",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_next = st.form_submit_button(
                "💾 保存して次の牛へ ⏩",
                use_container_width=True,
                type="primary",
            )
        with col_btn2:
            prev_btn = st.form_submit_button(
                "⬅️ 前の牛に戻る", use_container_width=True
            )

        if submit_next:
            if input_w is not None:
                st.session_state.cows_df.at[idx, "体重"] = float(input_w)
            if input_price is not None:
                st.session_state.cows_df.at[idx, "実際落札額(千円)"] = int(
                    input_price
                )
            st.session_state.cows_df.at[idx, "自社落札"] = input_purchased

            if idx + 1 < total_cows:
                st.session_state.current_no_idx = idx + 1
            st.rerun()

        if prev_btn:
            if idx > 0:
                st.session_state.current_no_idx = idx - 1
            st.rerun()

st.divider()

# --- 3. 一覧表エディタ & 判定結果 ---
st.subheader("📊 セリ一覧表 & 判定結果")

edited_df = st.data_editor(
    st.session_state.cows_df,
    num_rows="dynamic",
    use_container_width=True,
    height=280,
    column_config={
        "No": st.column_config.NumberColumn("No", format="%d", width="small"),
        "体重": st.column_config.NumberColumn(
            "当日体重(kg)", format="%.1f", width="small"
        ),
        "畜種": st.column_config.SelectboxColumn(
            "畜種",
            options=["F1", "黒毛", "褐毛", "JF", "H", "F1クロス"],
            width="small",
        ),
        "性別": st.column_config.SelectboxColumn(
            "性別", options=["去", "雌"], width="small"
        ),
        "日齢": st.column_config.NumberColumn(
            "日齢", format="%d", width="small"
        ),
        "実際落札額(千円)": st.column_config.NumberColumn(
            "実落札(千円)", format="%d", width="medium"
        ),
        "自社落札": st.column_config.CheckboxColumn(
            "自社落札?", default=False, width="small"
        ),
        "父": st.column_config.TextColumn("父", width="small"),
        "母の父": st.column_config.TextColumn("母の父", width="small"),
        "産次": st.column_config.NumberColumn(
            "産次", format="%d", width="small"
        ),
        "摘要": st.column_config.TextColumn("摘要", width="small"),
    },
)
st.session_state.cows_df = edited_df

# 計算ロジック適用
calc_results = edited_df.apply(calculate_cow, axis=1)
result_df = pd.concat([edited_df, calc_results], axis=1)


def judge(row):
    actual = row.get("実際落札額(千円)", 0)
    limit = row.get("上限価格(千円)", 0)
    if actual <= 0:
        return "-"
    diff = limit - actual
    return f"🟢 ＋{diff:,}千円 (得)" if diff >= 0 else f"🔴 {diff:,}千円 (高値)"


result_df["判定"] = result_df.apply(judge, axis=1)

st.dataframe(
    result_df[[
        "No",
        "自社落札",
        "上限価格(千円)",
        "実際落札額(千円)",
        "判定",
        "予測枝肉重量",
        "体重",
        "DG",
        "畜種",
        "性別",
        "父",
        "残育成日数",
        "育成コスト(千円)",
        "見込売上(千円)",
    ]],
    use_container_width=True,
    height=350,
)

st.divider()

# --- 4. kintoneへ一括送信ボタン ---
st.subheader("☁️ 牧場データ連携")
if st.button(
    "📤 セリ結果をkintoneに一括保存する",
    type="primary",
    use_container_width=True,
):
    with st.spinner("kintoneに詳細データを送信中..."):
        success, msg = send_to_kintone(result_df)
        if success:
            st.success(msg)
        else:
            st.error(msg)