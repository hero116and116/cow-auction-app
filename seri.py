import json
import os
import textwrap
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

# --- 牛のピクトグラム画像（Base64形式） ---
COW_ICON_B64 = (
    "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"
)

st.set_page_config(
    page_title="かう(セリのボーダー計算、結果保存アプリ)",
    page_icon="🐄",
    layout="centered",
)

# --- 通信切断検知バナー ---
_DISCONNECT_WATCHER_HTML = r"""
<script>
(function() {
  const doc = window.parent.document;
  const THRESHOLD_MS = 3 * 60 * 1000;

  if (!doc.getElementById('seri-disconnect-banner')) {
    const banner = doc.createElement('div');
    banner.id = 'seri-disconnect-banner';
    banner.style.cssText = [
      'display:none',
      'position:fixed',
      'top:0', 'left:0', 'right:0',
      'z-index:999999',
      'background-color:#f97316',
      'color:#ffffff',
      'padding:10px 14px',
      'font-size:14px',
      'font-weight:600',
      'text-align:center',
      'box-shadow:0 2px 6px rgba(0,0,0,0.25)',
      'align-items:center',
      'justify-content:center',
      'gap:10px',
      'flex-wrap:wrap'
    ].join(';');
    banner.innerHTML =
      '<span>\u26a0\ufe0f しばらく操作がなかったため、通信が切れている可能性があります</span>' +
      '<button id="seri-disconnect-reload" style="background:#ffffff;color:#c2410c;border:none;border-radius:4px;padding:6px 12px;font-weight:700;cursor:pointer;">\ud83d\udd04 今すぐ再読み込み</button>' +
      '<button id="seri-disconnect-close" style="background:transparent;color:#ffffff;border:none;font-size:18px;cursor:pointer;">\u2715</button>';
    doc.body.appendChild(banner);

    doc.getElementById('seri-disconnect-reload').addEventListener('click', function() {
      window.parent.location.reload();
    });
    doc.getElementById('seri-disconnect-close').addEventListener('click', function() {
      banner.style.display = 'none';
    });
  }

  if (!window.parent.__seriVisibilityWatcherAttached) {
    window.parent.__seriVisibilityWatcherAttached = true;
    window.parent.__seriHiddenAt = null;

    doc.addEventListener('visibilitychange', function() {
      if (doc.visibilityState === 'hidden') {
        window.parent.__seriHiddenAt = Date.now();
      } else if (doc.visibilityState === 'visible') {
        const hiddenAt = window.parent.__seriHiddenAt;
        if (hiddenAt && (Date.now() - hiddenAt) > THRESHOLD_MS) {
          const b = doc.getElementById('seri-disconnect-banner');
          if (b) {
            b.style.display = 'flex';
          }
        }
        window.parent.__seriHiddenAt = null;
      }
    });
  }
})();
</script>
"""
st_html(_DISCONNECT_WATCHER_HTML, height=0, width=0)

# --- マイナス要素の項目一覧 ---
NEGATIVE_FACTORS = [
    "馬面",
    "口が小さい",
    "尾枕がある",
    "皮膚の伸びが悪い",
    "背中が曲がっている",
]

# --- 自動バックアップ管理 ---
BACKUP_FILE = "backup_cows.json"


def save_backup():
  """現在のセリデータをJSONファイルに自動保存"""
  try:
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
      json.dump(st.session_state.cows, f, ensure_ascii=False, indent=2)
  except Exception:
    pass


def clear_backup():
  """バックアップファイルを削除（初期化用）"""
  if os.path.exists(BACKUP_FILE):
    try:
      os.remove(BACKUP_FILE)
    except Exception:
      pass


# --- カスタムCSS ---
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 480px; }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        position: static !important;
        background-color: #ffffff !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid="stTabs"] { position: static !important; }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        overflow: visible !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        white-space: nowrap !important;
    }

    /* 👇元の16pxに完全に戻しました（画面2への影響をなくすため） */
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
        box-sizing: border-box;
    }
    .card-divider {
        border-top: 2px solid #1e293b;
        margin: 0;
    }

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

    /* 落札価格入力画面：3カラム横並び */
    .cow-top-row {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 2px;
        text-align: left;
    }
    .cow-info-col {
        flex: 0 0 auto;
        min-width: 0;
    }
    .cow-top-row .cow-meta {
        margin-top: 0;
        font-size: 13px;
        line-height: 1.45;
        padding-right: 4px;
    }
    .cow-icon-col {
        flex: 0 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-left: 2px;
    }
    .cow-top-row .cow-icon-container {
        width: 120px;
        height: 78px;
        margin: 0;
        flex-shrink: 0;
    }
    .cow-top-row .cow-number-overlay {
        font-size: 19px;
    }
    .cow-neg-col {
        flex: 0 0 62px;
        min-width: 0;
        margin-left: auto;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 3px;
    }

    .input-display-row {
        display: flex;
        align-items: flex-end;
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
    .cow-metrics .target-price { color: #dc2626; font-size: 16px; }

    .neg-badge {
        display: inline-block;
        background-color: #fff7ed;
        color: #c2410c;
        border: 1px solid #fdba74;
        border-radius: 6px;
        font-size: 9.5px;
        font-weight: 700;
        padding: 1px 5px;
        line-height: 1.3;
        white-space: normal;
        word-break: keep-all;
    }

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

    .st-key-negative_factors_area_w {
        border: 2px solid #1e293b;
        border-top: none;
        margin-top: -16px;
        padding: 6px 12px;
        background-color: #ffffff;
        height: 40px;
        box-sizing: border-box;
        overflow: hidden;
    }
    .st-key-negative_factors_area_w [data-testid="stPills"] {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        flex-wrap: nowrap !important;
        overflow-x: auto;
        overflow-y: hidden;
        gap: 5px;
        height: 100%;
        -webkit-overflow-scrolling: touch;
    }
    .st-key-negative_factors_area_w [data-testid="stPills"] button {
        font-size: 11px !important;
        padding: 1px 9px !important;
        min-height: 26px !important;
        height: 26px !important;
        flex-shrink: 0 !important;
        white-space: nowrap !important;
    }

    .st-key-numpad_area_w, .st-key-numpad_area_p {
        border: 2px solid #1e293b;
        border-top: none;
        border-radius: 0 0 4px 4px;
        margin-top: -16px;
        padding: 12px 6px 16px 6px;
        background-color: #ffffff;
    }

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

    .st-key-btn_w_enter button, .st-key-btn_p_enter button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
    }
    .st-key-btn_w_c button, .st-key-btn_p_c button {
        background-color: #f1f5f9 !important;
        color: #dc2626 !important;
    }

    .st-key-numpad_area_w div[data-testid="element-container"],
    .st-key-numpad_area_p div[data-testid="element-container"] {
        margin: 0 !important;
    }
    .st-key-numpad_area_w div[data-testid="element-container"]:not(:last-child),
    .st-key-numpad_area_p div[data-testid="element-container"]:not(:last-child) {
        margin-bottom: 6px !important;
    }

    .st-key-prev_w button, .st-key-prev_p button {
        height: 306px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        margin-top: 0 !important;
    }

    .st-key-ce_w button, .st-key-ce_p button {
        height: 72px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #fff7ed !important;
        color: #c2410c !important;
        padding: 0 !important;
    }
    .st-key-ce_w button:hover, .st-key-ce_p button:hover {
        border-color: #3b82f6 !important;
        color: #3b82f6 !important;
    }
    .st-key-next_w button, .st-key-next_p button {
        height: 228px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        margin-top: 0 !important;
    }
    
    /* 🔴 スマホでの「額／番号」ボタン横並び強制設定と隙間埋め */
    /* 👇 マイナスマージンを調整し、文字1つ分（約20px）下に下げました */
    .st-key-mode_switch_area {
        margin-top: -26px !important; 
    }
    .st-key-mode_switch_area div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
    }
    .st-key-mode_switch_area div[data-testid="stColumn"],
    .st-key-mode_switch_area div[data-testid="column"] {
        width: 50% !important;
        min-width: 0 !important;
    }
    /* ボタンの枠のサイズ設定 */
    .st-key-btn_switch_price button, .st-key-btn_switch_buyer button {
        height: 70px !important; /* 文字に合わせて枠も少し大きくしました */
        background-color: #f8fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #475569 !important;
        padding: 4px !important;
        margin-bottom: 8px !important;
    }
    
    /* 👇 ボタンの中の「文字」を直接指定して強制的に大きくする */
    .st-key-btn_switch_price button p, .st-key-btn_switch_buyer button p {
        font-size: 30px !important;  /* ここで文字の大きさを変えられます */
        font-weight: 900 !important;
        margin: 0 !important;
    }
    
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
""",
    unsafe_allow_html=True,
)

# --- APIキー & kintone設定 ---
GEMINI_API_KEY = "AQ.Ab8RN6KGaI97aQ0liR_8kYw5ALr-SMS8KzDW8cPaMUnlt4veDQ"
KINTONE_DOMAIN = "cattlook.cybozu.com"
KINTONE_APP_ID = "131"
KINTONE_API_TOKEN = "T4aTJyzRN736eaqzzWucxZIIbXy9wYn5YkAnlJsO"

# --- サイドバー開閉状態 ---
if "sidebar_open" not in st.session_state:
  st.session_state.sidebar_open = False

if not st.session_state.sidebar_open:
  st.markdown(
      "<style>section[data-testid='stSidebar']{display:none"
      " !important;}</style>",
      unsafe_allow_html=True,
  )
  if st.button("⚙️", key="open_sidebar_btn"):
    st.session_state.sidebar_open = True
    st.rerun()

# --- サイドバー設定 ---
if "settings" not in st.session_state:
  st.session_state.settings = {
      "carcass_price": 2300,
      "daily_cost": 850,
      "shipment_days": 854,
      "birth_weight": 35.0,
      "yield_rate_mc": 0.645,
      "yield_rate_f": 0.625,
      "target_profit": 100,
  }

with st.sidebar:
  if st.button("✕ 閉じる", key="close_sidebar_btn", use_container_width=True):
    st.session_state.sidebar_open = False
    st.rerun()
  st.header("⚙️ 共通設定（相場・コスト）")
  input_carcass_price = st.number_input(
      "枝肉単価 (円/kg)",
      value=st.session_state.settings["carcass_price"],
      step=50,
      key="input_carcass_price",
  )
  input_daily_cost = st.number_input(
      "1日あたり育成コスト (円)",
      value=st.session_state.settings["daily_cost"],
      step=10,
      key="input_daily_cost",
  )
  input_shipment_days = st.number_input(
      "出荷日齢 (日)",
      value=st.session_state.settings["shipment_days"],
      step=1,
      key="input_shipment_days",
  )
  input_birth_weight = st.number_input(
      "生時体重 (kg)",
      value=st.session_state.settings["birth_weight"],
      step=1.0,
      key="input_birth_weight",
  )
  input_yield_rate_mc = st.number_input(
      "歩留基準【去勢】",
      value=st.session_state.settings.get("yield_rate_mc", 0.645),
      step=0.005,
      format="%.3f",
      key="input_yield_rate_mc",
  )
  input_yield_rate_f = st.number_input(
      "歩留基準【雌】",
      value=st.session_state.settings.get("yield_rate_f", 0.625),
      step=0.005,
      format="%.3f",
      key="input_yield_rate_f",
  )
  input_target_profit = st.number_input(
      "目標利益 (千円)",
      value=st.session_state.settings["target_profit"],
      step=10,
      key="input_target_profit",
  )

  if st.button("💾 設定を保存", use_container_width=True, type="primary"):
    st.session_state.settings = {
        "carcass_price": input_carcass_price,
        "daily_cost": input_daily_cost,
        "shipment_days": input_shipment_days,
        "birth_weight": input_birth_weight,
        "yield_rate_mc": input_yield_rate_mc,
        "yield_rate_f": input_yield_rate_f,
        "target_profit": input_target_profit,
    }
    st.session_state.metrics_dirty = True
    st.toast("設定を保存し、再計算しました ✅", icon="💾")
    st.rerun()

  st.divider()
  if st.button("🗑️ 作業データを全初期化", use_container_width=True):
    clear_backup()
    next_reset_ver = st.session_state.get("reset_ver", 0) + 1
    st.session_state.clear()
    st.session_state.reset_ver = next_reset_ver
    st.rerun()


# --- 成長曲線パラメータ ---
GROWTH_CURVE_PARAMS = {
    "雌": {"A": 734.258662, "B": 9.925742, "k": 0.005825},
    "去": {"A": 791.207131, "B": 9.810946, "k": 0.005967},
    "全体": {"A": 739.731502, "B": 10.204743, "k": 0.005949},
}


def logistic_weight(t, A, B, k):
  return A / (1 + B * np.exp(-k * t))


def logistic_dg(t, A, B, k):
  return (A * B * k * np.exp(-k * t)) / ((1 + B * np.exp(-k * t)) ** 2)


def get_growth_params(gender):
  return GROWTH_CURVE_PARAMS.get(gender, GROWTH_CURVE_PARAMS["全体"])


# --- 計算ロジック ---
def calculate_cow_metrics(cow_row):
  settings = st.session_state.settings
  carcass_price = settings["carcass_price"]
  daily_cost = settings["daily_cost"]
  shipment_days = settings["shipment_days"]
  birth_weight = settings["birth_weight"]

  gender_clean = clean_gender(cow_row.get("性別", "去"))
  yield_rate = (
      settings.get("yield_rate_f", 0.625)
      if gender_clean == "雌"
      else settings.get("yield_rate_mc", 0.645)
  )
  target_profit = settings["target_profit"]

  try:
    days = float(cow_row.get("日齢", 0))
    weight = float(cow_row.get("体重", 0))
  except (ValueError, TypeError):
    days, weight = 0, 0

  if days <= 0 or weight <= birth_weight:
    return {
        "DG": 0.0,
        "育成日数": 0,
        "育成コスト": 0,
        "予測出荷体重": 0.0,
        "予測枝肉重量": 0.0,
        "見込売上": 0,
        "ボーダー価格": 0,
        "目標落札額": 0,
    }

  dg = (weight - birth_weight) / days
  raising_days = max(0, shipment_days - days)
  cost = int(raising_days * daily_cost)

  params = get_growth_params(gender_clean)
  A, B, k = params["A"], params["B"], params["k"]

  offset = weight - logistic_weight(days, A, B, k)
  pred_ship_weight = max(weight, logistic_weight(shipment_days, A, B, k) + offset)

  pred_carcass_weight = pred_ship_weight * yield_rate
  sales = int(pred_carcass_weight * carcass_price)
  border_price = max(0, (sales - cost) // 1000)
  target_price = max(0, border_price - target_profit)

  return {
      "DG": round(dg, 3),
      "育成日数": int(raising_days),
      "育成コスト": cost // 1000,
      "予測出荷体重": round(pred_ship_weight, 1),
      "予測枝肉重量": round(pred_carcass_weight, 1),
      "見込売上": sales // 1000,
      "ボーダー価格": border_price,
      "目標落札額": target_price,
  }


# --- kg単価の平均を計算する関数（累計変数による高速化 ＋ 0円除外） ---
def calculate_average_unit_price():
  """
  落札額が確定している牛たちのkg単価の平均を算出する。
  実際落札額が0以下の牛は自動で除外する。
  """
  if "avg_unit_price_cache" not in st.session_state or st.session_state.get("metrics_dirty", True):
    total_sum = 0
    count = 0
    for c in st.session_state.cows:
      price = c.get("実際落札額", 0)
      weight = c.get("体重", 0)
      if price and price > 0 and weight and weight > 0:
        kp = int(round(price * 1000 / weight))
        total_sum += kp
        count += 1
    
    st.session_state.total_unit_price_sum = total_sum
    st.session_state.bid_count = count
    st.session_state.avg_unit_price_cache = int(round(total_sum / count)) if count > 0 else 0
  
  return st.session_state.avg_unit_price_cache


def clean_gender(val):
  s = str(val).strip()
  return "雌" if ("雌" in s or "メス" in s or "めす" in s) else "去"


# --- Gemini 名簿解析 ---
def parse_catalog_file(uploaded_file, key=GEMINI_API_KEY):
  client = genai.Client(api_key=key)
  file_bytes = uploaded_file.getvalue()
  mime_type = (
      "application/pdf"
      if uploaded_file.name.lower().endswith(".pdf")
      else "image/jpeg"
  )

  prompt = """
    添付された牛のセリ名簿から各行の情報を抽出し、JSON配列として出力してください。
    キー: No (整数), 個体識別番号 (文字列、10桁の数字、名簿の表記そのまま), 性別 (去/雌), 生年月日 (文字列、名簿の表記そのまま。例: R07.11.08), 日齢 (整数), 産次 (整数、無ければ0), 摘要 (文字列), 父 (文字列), 母の父 (文字列), 母の祖父 (文字列), 母の母の祖父 (文字列)
    ※ 体重・落札額は0にしてください。個体識別番号が名簿に無い場合は空文字にしてください。JSON配列のみを出力してください。
    """
  response = client.models.generate_content(
      model="gemini-3.6-flash",
      contents=[
          types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
          prompt,
      ],
      config=types.GenerateContentConfig(response_mime_type="application/json"),
  )
  data = json.loads(response.text)
  for r in data:
    r["性別"] = clean_gender(r.get("性別", "去"))
    r["個体識別番号"] = str(r.get("個体識別番号", "") or "")
    r["落札者番号"] = ""
  return data


# --- kintone 送信 ---
def send_to_kintone(cows_list):
  url = f"https://{KINTONE_DOMAIN}/k/v1/records.json"
  headers = {
      "X-Cybozu-API-Token": KINTONE_API_TOKEN,
      "Content-Type": "application/json",
  }

  records = []
  settings = st.session_state.settings
  for c in cows_list:
    weight = float(c.get("体重", 0))
    price = int(c.get("実際落札額", 0))
    if weight > 0 or price > 0:
      status = "購入" if c.get("自社落札", False) else "非購入"
      calc = calculate_cow_metrics(c)
      negs = set(c.get("マイナス要素", []))
      records.append({
          "出場番号": {"value": int(c["No"])},
          "個体識別番号": {"value": str(c.get("個体識別番号", ""))},
          "当日体重": {"value": weight},
          "日齢": {"value": int(c.get("日齢", 0))},
          "産次": {"value": int(c.get("産次", 0))},
          "摘要": {"value": str(c.get("摘要", ""))},
          "性別": {"value": c.get("性別", "去")},
          "父": {"value": str(c.get("父", ""))},
          "母の父": {"value": str(c.get("母の父", ""))},
          "母の祖父": {"value": str(c.get("母の祖父", ""))},
          "母の母の祖父": {"value": str(c.get("母の母の祖父", ""))},
          "馬面": {"value": ["該当"] if "馬面" in negs else []},
          "口が小さい": {"value": ["該当"] if "口が小さい" in negs else []},
          "尾枕がある": {"value": ["該当"] if "尾枕がある" in negs else []},
          "皮膚の伸びが悪い": {"value": ["該当"] if "皮膚の伸びが悪い" in negs else []},
          "背中が曲がっている": {"value": ["該当"] if "背中が曲がっている" in negs else []},
          "落札上限価格": {"value": calc["ボーダー価格"]},
          "予測出荷体重": {"value": calc["予測出荷体重"]},
          "実際落札額": {"value": price},
          "購入結果": {"value": status},
          # ※ キントーン側に「落札者番号」フィールドを追加した後に、以下の行のコメントアウトを外して有効化してください
          # "落札者番号": {"value": str(c.get("落札者番号", ""))},
          "設定枝肉単価": {"value": settings["carcass_price"]},
          "設定育成コスト": {"value": settings["daily_cost"]},
          "設定出荷日齢": {"value": settings["shipment_days"]},
          "設定生時体重": {"value": settings["birth_weight"]},
          "設定歩留基準_去勢": {"value": settings["yield_rate_mc"]},
          "設定歩留基準_雌": {"value": settings["yield_rate_f"]},
          "設定目標利益": {"value": settings["target_profit"]},
      })
  if not records:
    return False, "送信対象のデータがありません。"
  res = requests.post(
      url, headers=headers, json={"app": KINTONE_APP_ID, "records": records}
  )
  return (
      (True, f"✅ {len(records)} 頭のデータをkintoneに保存しました！")
      if res.status_code == 200
      else (False, f"❌ エラー: {res.text}")
  )


# --- セッションステート初期化 ---
if "cows" not in st.session_state:
  if os.path.exists(BACKUP_FILE):
    try:
      with open(BACKUP_FILE, "r", encoding="utf-8") as f:
        st.session_state.cows = json.load(f)
        # 過去データに落札者番号がない場合の補完
        for c in st.session_state.cows:
            if "落札者番号" not in c:
                c["落札者番号"] = ""
      st.toast("前回の作業データを復元しました 📦", icon="✅")
    except Exception:
      st.session_state.cows = [{
          "No": i,
          "個体識別番号": "",
          "体重": 0,
          "実際落札額": 0,
          "落札者番号": "",
          "性別": "去",
          "生年月日": "-",
          "日齢": 280,
          "産次": 1,
          "父": "福勝鶴",
          "母の父": "美津照重",
          "母の祖父": "平茂勝",
          "母の母の祖父": "-",
          "摘要": "",
          "自社落札": False,
          "マイナス要素": [],
      } for i in range(1, 31)]
  else:
    st.session_state.cows = [{
        "No": i,
        "個体識別番号": "",
        "体重": 0,
        "実際落札額": 0,
        "落札者番号": "",
        "性別": "去",
        "生年月日": "-",
        "日齢": 280,
        "産次": 1,
        "父": "福勝鶴",
        "母の父": "美津照重",
        "母の祖父": "平茂勝",
        "母の母の祖父": "-",
        "摘要": "",
        "自社落札": False,
        "マイナス要素": [],
    } for i in range(1, 31)]

if "curr_idx_w" not in st.session_state:
  st.session_state.curr_idx_w = 0
if "curr_idx_p" not in st.session_state:
  st.session_state.curr_idx_p = 0
if "input_buffer_w" not in st.session_state:
  st.session_state.input_buffer_w = ""
if "input_buffer_p" not in st.session_state:
  st.session_state.input_buffer_p = ""
if "p_mode" not in st.session_state: # 落札入力画面のモード管理 ("price" or "buyer")
  st.session_state.p_mode = "price"
if "avg_unit_price_cache" not in st.session_state:
  st.session_state.avg_unit_price_cache = 0
if "total_unit_price_sum" not in st.session_state:
  st.session_state.total_unit_price_sum = 0
if "bid_count" not in st.session_state:
  st.session_state.bid_count = 0
if "metrics_dirty" not in st.session_state:
  st.session_state.metrics_dirty = True
if "reset_ver" not in st.session_state:
  st.session_state.reset_ver = 0


# --- 牛のピクトグラムヘルパー ---
def get_cow_svg(number_str):
  html = f"""<div class="cow-icon-container">
<img class="cow-img" src="data:image/png;base64,{COW_ICON_B64}" alt="cow"/>
<div class="cow-number-overlay">{number_str}</div>
</div>"""
  return html


# --- メインタブ（5タブ構成） ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 事前データ自動読み取り",
    "⚖️ 体重入力（下見）",
    "🎯 落札価格入力（本番）",
    "📈 kg単価推移グラフ",
    "📊 セリ結果一覧",
])

# =========================================================
# 画面1: 事前データ自動読み取り画面 ＋ 手動追加機能
# =========================================================
with tab1:
  st.subheader("📄 セリ名簿の自動読み取り")

  if st.session_state.get("just_parsed_count"):
    st.success(
        "✅ 読み取りが完了しました！（"
        f"{st.session_state.just_parsed_count}頭のデータを読み込みました）"
    )
    st.session_state.just_parsed_count = 0

  uploaded = st.file_uploader(
      "名簿ファイルまたは写真を選択", type=["pdf", "png", "jpg", "jpeg"]
  )
  if uploaded and st.button(
      "🚀 自動読み取り開始", type="primary", use_container_width=True
  ):
    with st.spinner("AIが名簿を解析中..."):
      try:
        parsed = parse_catalog_file(uploaded)
        if parsed:
          for r in parsed:
            r["体重"] = 0
            r["実際落札額"] = 0
            r["落札者番号"] = ""
            r["自社落札"] = False
            r["マイナス要素"] = []
          st.session_state.cows = parsed
          st.session_state.metrics_dirty = True
          st.session_state.curr_idx_w = 0
          st.session_state.curr_idx_p = 0
          st.session_state.input_buffer_w = ""
          st.session_state.input_buffer_p = ""
          st.session_state.p_mode = "price"
          st.session_state.just_parsed_count = len(parsed)
          save_backup()
          st.toast("読み取りが完了しました！", icon="✅")
          st.rerun()
      except Exception as e:
        st.error(f"❌ AIの読み取り中にサーバーエラーが発生しました。しばらく待ってから再度お試しください。\n\n詳細: {e}")

  st.divider()

  # --- ➕ 急遽追加された牛を手動で登録するフォーム ---
  st.subheader("➕ 牛の手動追加（追加・修正用）")
  st.markdown("名簿にない牛が急遽追加された場合や、情報を手動で追加したい場合に登録できます。")

  with st.form("manual_add_cow_form"):
    add_no = st.number_input("出場番号 (No)", min_value=1, step=1, value=int(max([c["No"] for c in st.session_state.cows], default=0) + 1))
    add_gender = st.selectbox("性別", ["去", "雌"])
    add_days = st.number_input("日齢", min_value=1, step=1, value=280)
    add_weight = st.number_input("当日体重 (kg)", min_value=0.0, step=1.0, value=0.0)
    add_father = st.text_input("父牛", value="")
    add_memo = st.text_input("摘要（メモなど）", value="")

    submitted = st.form_submit_button("➕ この牛を追加する", use_container_width=True)
    if submitted:
      # すでに同じNoの牛がいないか確認、あれば上書き、なければ追加
      existing_cow = next((c for c in st.session_state.cows if c["No"] == add_no), None)
      if existing_cow:
        existing_cow["性別"] = add_gender
        existing_cow["日齢"] = int(add_days)
        existing_cow["体重"] = float(add_weight)
        existing_cow["父"] = add_father
        existing_cow["摘要"] = add_memo
        st.toast(f"No.{add_no} の情報を更新しました！", icon="✅")
      else:
        new_cow = {
            "No": int(add_no),
            "個体識別番号": "",
            "体重": float(add_weight),
            "実際落札額": 0,
            "落札者番号": "",
            "性別": add_gender,
            "生年月日": "-",
            "日齢": int(add_days),
            "産次": 1,
            "父": add_father if add_father else "-",
            "母の父": "-",
            "母の祖父": "-",
            "母の母の祖父": "-",
            "摘要": add_memo,
            "自社落札": False,
            "マイナス要素": [],
        }
        st.session_state.cows.append(new_cow)
        # 出場番号順（No順）にリストを綺麗にソート
        st.session_state.cows = sorted(st.session_state.cows, key=lambda x: x["No"])
        st.toast(f"No.{add_no} を新しく追加しました！", icon="🎉")
      
      st.session_state.metrics_dirty = True
      save_backup()
      st.rerun()


# =========================================================
# 画面2: 体重入力画面（下見）
# =========================================================
with tab2:
  total = len(st.session_state.cows)
  idx = st.session_state.curr_idx_w
  cow = st.session_state.cows[idx]

  with st.container(key="no_jump_w"):
    with st.popover(f"No.{cow['No']} ✎"):
      nos = [c["No"] for c in st.session_state.cows]
      target_no = st.number_input(
          "出場番号を入力して移動",
          min_value=int(min(nos)),
          max_value=int(max(nos)),
          value=int(cow["No"]),
          step=1,
          key=f"jump_no_w_{idx}",
      )
      if st.button(
          "この番号へ移動", key=f"jump_go_w_{idx}", use_container_width=True
      ):
        target_idx = next(
            (
                i
                for i, c in enumerate(st.session_state.cows)
                if c["No"] == target_no
            ),
            None,
        )
        if target_idx is not None:
          if st.session_state.input_buffer_w:
            st.session_state.cows[idx]["体重"] = float(
                st.session_state.input_buffer_w
            )
            st.session_state.metrics_dirty = True
            save_backup()
          st.session_state.curr_idx_w = target_idx
          st.session_state.input_buffer_w = ""
          st.rerun()
        else:
          st.warning("その出場番号は見つかりませんでした。")

  display_w = (
      st.session_state.input_buffer_w
      if st.session_state.input_buffer_w != ""
      else (str(cow["体重"]) if cow["体重"] > 0 else "")
  )

  card_html_w = (
      '<div class="screen-card">'
      '<div class="card-top">'
      f'{get_cow_svg(cow["No"])}'
      '<div class="input-display-row">'
      f'<span class="input-display">{display_w}</span>'
      '<span class="input-unit">kg</span>'
      "</div>"
      '<div class="cow-meta">'
      f'性別: <b>{cow["性別"]}</b> ｜ 日齢: <b>{cow["日齢"]}日</b> ｜ 父:'
      f' <b>{cow["父"]}</b>'
      "</div>"
      "</div>"
      "</div>"
  )
  st.markdown(card_html_w, unsafe_allow_html=True)

  with st.container(key="negative_factors_area_w"):
    current_negs = cow.get("マイナス要素", [])
    selected_negs = st.pills(
        "マイナス要素（該当する場合のみ選択）",
        options=NEGATIVE_FACTORS,
        selection_mode="multi",
        default=[f for f in current_negs if f in NEGATIVE_FACTORS],
        key=f"neg_pills_{st.session_state.reset_ver}_{idx}",
        label_visibility="collapsed",
    )
    new_negs = (
        [f for f in NEGATIVE_FACTORS if f in selected_negs]
        if selected_negs
        else []
    )
    if set(new_negs) != set(current_negs):
      st.session_state.cows[idx]["マイナス要素"] = new_negs
      save_backup()

  with st.container(key="numpad_area_w"):
    col_l, col_pad, col_r = st.columns([0.8, 4.6, 0.8])

    with col_l:
      if st.button("←", key="prev_w", use_container_width=True):
        if st.session_state.input_buffer_w:
          st.session_state.cows[idx]["体重"] = float(
              st.session_state.input_buffer_w
          )
          st.session_state.metrics_dirty = True
          save_backup()
        st.session_state.curr_idx_w = max(0, idx - 1)
        st.session_state.input_buffer_w = ""
        st.rerun()

    with col_r:
      with st.container(key="numpad_right_w"):
        if st.button("BS", key="ce_w", use_container_width=True):
          st.session_state.input_buffer_w = st.session_state.input_buffer_w[:-1]
          st.rerun()

        if st.button("→", key="next_w", use_container_width=True):
          if st.session_state.input_buffer_w:
            st.session_state.cows[idx]["体重"] = float(
                st.session_state.input_buffer_w
            )
            st.session_state.metrics_dirty = True
            save_backup()
          st.session_state.curr_idx_w = min(total - 1, idx + 1)
          st.session_state.input_buffer_w = ""
          st.rerun()

    with col_pad:
      for row_nums in [["7", "8", "9"], ["4", "5", "6"], ["1", "2", "3"]]:
        cols = st.columns(3)
        for i, num in enumerate(row_nums):
          if cols[i].button(num, key=f"btn_w_{num}", use_container_width=True):
            st.session_state.input_buffer_w += num
            st.rerun()
      cols_bottom = st.columns(3)
      if cols_bottom[0].button("C", key="btn_w_c", use_container_width=True):
        st.session_state.input_buffer_w = ""
        st.session_state.cows[idx]["体重"] = 0
        st.session_state.metrics_dirty = True
        save_backup()
        st.rerun()
      if cols_bottom[1].button("0", key="btn_w_0", use_container_width=True):
        st.session_state.input_buffer_w += "0"
        st.rerun()
      if cols_bottom[2].button(
          "決定", key="btn_w_enter", use_container_width=True
      ):
        if st.session_state.input_buffer_w:
          st.session_state.cows[idx]["体重"] = float(
              st.session_state.input_buffer_w
          )
          st.session_state.metrics_dirty = True
          save_backup()
        st.session_state.input_buffer_w = ""
        st.rerun()


# =========================================================
# 画面3: 落札価格入力画面（セリ本番）
# =========================================================
with tab3:
  total = len(st.session_state.cows)
  idx = st.session_state.curr_idx_p
  cow = st.session_state.cows[idx]

  # kg単価の平均値を取得（累計変数による高速化 ＋ 0円自動除外）
  avg_unit_price = calculate_average_unit_price()

  calc = calculate_cow_metrics(cow)

  # モードに応じた表示の切り替えと、未入力時のハイフン処理
  if st.session_state.p_mode == "price":
      display_p = st.session_state.input_buffer_p if st.session_state.input_buffer_p != "" else (str(cow["実際落札額"]) if cow["実際落札額"] > 0 else "-")
      display_b = str(cow.get("落札者番号", "")) if str(cow.get("落札者番号", "")) != "" else "-"
      active_price_class = "active-input"
      active_buyer_class = ""
  else:
      display_p = str(cow["実際落札額"]) if cow["実際落札額"] > 0 else "-"
      display_b = st.session_state.input_buffer_p if st.session_state.input_buffer_p != "" else (str(cow.get("落札者番号", "")) if str(cow.get("落札者番号", "")) != "" else "-")
      active_price_class = ""
      active_buyer_class = "active-input"

  neg_factors = [f for f in NEGATIVE_FACTORS if f in cow.get("マイナス要素", [])]
  neg_badges_html = "".join(
      f'<span class="neg-badge">{n}</span>' for n in neg_factors
  )

  with st.container(key="no_jump_p"):
    with st.popover(f"No.{cow['No']} ✎"):
      nos = [c["No"] for c in st.session_state.cows]
      target_no = st.number_input(
          "出場番号を入力して移動",
          min_value=int(min(nos)),
          max_value=int(max(nos)),
          value=int(cow["No"]),
          step=1,
          key=f"jump_no_p_{idx}",
      )
      if st.button(
          "この番号へ移動", key=f"jump_go_p_{idx}", use_container_width=True
      ):
        target_idx = next(
            (
                i
                for i, c in enumerate(st.session_state.cows)
                if c["No"] == target_no
            ),
            None,
        )
        if target_idx is not None:
          # 移動前に今の状態を保存してモードをリセット
          if st.session_state.input_buffer_p:
            if st.session_state.p_mode == "price":
                st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer_p)
                st.session_state.metrics_dirty = True
            else:
                st.session_state.cows[idx]["落札者番号"] = st.session_state.input_buffer_p
            save_backup()
          st.session_state.p_mode = "price"
          st.session_state.curr_idx_p = target_idx
          st.session_state.input_buffer_p = ""
          st.rerun()
        else:
          st.warning("その出場番号は見つかりませんでした。")

  try:
    price_for_unit = int(cow["実際落札額"])
  except (ValueError, TypeError):
    price_for_unit = 0
  try:
    weight_for_unit = float(cow["体重"])
  except (ValueError, TypeError):
    weight_for_unit = 0
  unit_price = (
      int(round(price_for_unit * 1000 / weight_for_unit))
      if weight_for_unit > 0 and price_for_unit > 0
      else 0
  )
  unit_price_text = f"{unit_price:,}円/kg" if unit_price > 0 else "-"

  # 画面3専用のインラインスタイルで下の余白をゼロにする
  card_html_p = (
      '<div class="screen-card" style="margin-bottom: 0px !important;">'
      '<div class="card-top">'
      '<div class="cow-top-row">'
      '<div class="cow-info-col cow-meta">'
      f'日齢: <b>{cow["日齢"]}日</b><br>'
      f'体重: <b>{cow["体重"]}kg</b><br>'
      f'父: <b>{cow["父"]}</b><br>'
      f'DG: <b>{calc["DG"]}kg/日</b><br>'
      f"kg単価:<br><b>{unit_price_text}</b><br>"
      f'摘要: <b>{cow.get("摘要", "") or "-"}</b>'
      "</div>"
      f'<div class="cow-icon-col">{get_cow_svg(cow["No"])}</div>'
      f'<div class="cow-neg-col">{neg_badges_html}</div>'
      "</div>"
      '<div class="cow-metrics">'
      f'落札牛平均kg単価 <span class="profit">{avg_unit_price:,}</span>(円/kg)<br>'
      f'損益分岐点 <span class="border-price">{calc["ボーダー価格"]}</span>(千円)<br>'
      f'目標落札額 <span class="target-price">{calc["目標落札額"]}</span>(千円)'
      "</div>"
      "</div>"
      "</div>"
  )
  st.markdown(card_html_p, unsafe_allow_html=True)

  # --- 🔴 画面3専用：アクティブな入力モードを強調する動的CSS ---
  active_css = f"""
  <style>
  .st-key-btn_switch_{st.session_state.p_mode} button {{
      border: 2px solid #f97316 !important;
      border-bottom: 4px solid #f97316 !important;
      background-color: #fff7ed !important;
      color: #ea580c !important;
  }}
  </style>
  """
  st.markdown(active_css, unsafe_allow_html=True)

  # --- タップで切り替え可能な新しい入力エリア ---
  with st.container(key="mode_switch_area"):
      col_p, col_b = st.columns(2)
      with col_p:
          if st.button(f"額: {display_p} 千円", key="btn_switch_price", use_container_width=True):
              if st.session_state.p_mode != "price":
                  # 購買者番号モードから戻る際、入力途中のデータがあれば保存
                  if st.session_state.input_buffer_p:
                      st.session_state.cows[idx]["落札者番号"] = st.session_state.input_buffer_p
                      save_backup()
                  st.session_state.input_buffer_p = ""
                  st.session_state.p_mode = "price"
                  st.rerun()

      with col_b:
          if st.button(f"番号: {display_b}", key="btn_switch_buyer", use_container_width=True):
              if st.session_state.p_mode != "buyer":
                  # 落札額モードから切り替える際、入力途中のデータがあれば保存
                  if st.session_state.input_buffer_p:
                      st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer_p)
                      st.session_state.metrics_dirty = True
                      save_backup()
                  st.session_state.input_buffer_p = ""
                  st.session_state.p_mode = "buyer"
                  st.rerun()

  with st.container(key="purchase_check_area_p"):
    purchased = st.checkbox(
        "購入チェック",
        value=cow["自社落札"],
        key=f"buy_check_{st.session_state.reset_ver}_{idx}",
    )
    if purchased != cow["自社落札"]:
      st.session_state.cows[idx]["自社落札"] = purchased
      st.session_state.metrics_dirty = True
      save_backup()

  with st.container(key="numpad_area_p"):
    col_l, col_pad, col_r = st.columns([0.8, 4.6, 0.8])

    with col_l:
      if st.button("←", key="prev_p", use_container_width=True):
        if st.session_state.input_buffer_p:
          if st.session_state.p_mode == "price":
              st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer_p)
              st.session_state.metrics_dirty = True
          else:
              st.session_state.cows[idx]["落札者番号"] = st.session_state.input_buffer_p
          save_backup()
        st.session_state.curr_idx_p = max(0, idx - 1)
        st.session_state.input_buffer_p = ""
        st.session_state.p_mode = "price" # 常に落札額入力からスタート
        st.rerun()

    with col_r:
      with st.container(key="numpad_right_p"):
        if st.button("BS", key="ce_p", use_container_width=True):
          st.session_state.input_buffer_p = st.session_state.input_buffer_p[:-1]
          st.rerun()

        if st.button("→", key="next_p", use_container_width=True):
          if st.session_state.input_buffer_p:
            if st.session_state.p_mode == "price":
                st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer_p)
                st.session_state.metrics_dirty = True
            else:
                st.session_state.cows[idx]["落札者番号"] = st.session_state.input_buffer_p
            save_backup()
          st.session_state.curr_idx_p = min(total - 1, idx + 1)
          st.session_state.input_buffer_p = ""
          st.session_state.p_mode = "price" # 常に落札額入力からスタート
          st.rerun()

    with col_pad:
      for row_nums in [["7", "8", "9"], ["4", "5", "6"], ["1", "2", "3"]]:
        cols = st.columns(3)
        for i, num in enumerate(row_nums):
          if cols[i].button(num, key=f"btn_p_{num}", use_container_width=True):
            st.session_state.input_buffer_p += num
            st.rerun()
      cols_bottom = st.columns(3)
      if cols_bottom[0].button("C", key="btn_p_c", use_container_width=True):
        st.session_state.input_buffer_p = ""
        if st.session_state.p_mode == "price":
            st.session_state.cows[idx]["実際落札額"] = 0
            st.session_state.metrics_dirty = True
        else:
            st.session_state.cows[idx]["落札者番号"] = ""
        save_backup()
        st.rerun()
      if cols_bottom[1].button("0", key="btn_p_0", use_container_width=True):
        st.session_state.input_buffer_p += "0"
        st.rerun()
      
      if cols_bottom[2].button("決定", key="btn_p_enter", use_container_width=True):
        if st.session_state.p_mode == "price":
            if st.session_state.input_buffer_p:
                st.session_state.cows[idx]["実際落札額"] = int(st.session_state.input_buffer_p)
                st.session_state.metrics_dirty = True
                save_backup()
            st.session_state.input_buffer_p = ""
            st.session_state.p_mode = "buyer" # 決定後、落札者番号モードへ移行
            st.rerun()
        else: # buyerモードの場合
            if st.session_state.input_buffer_p:
                st.session_state.cows[idx]["落札者番号"] = st.session_state.input_buffer_p
                save_backup()
            st.session_state.input_buffer_p = ""
            st.session_state.p_mode = "price" # モードをリセットする（遷移はしない）
            st.rerun()


# =========================================================
# 画面4: kg単価推移グラフ画面（正しい番号順 ＆ Y軸1000〜3500固定）
# =========================================================
with tab4:
  st.subheader("📈 kg単価の推移グラフ")
  if st.button("🔄 グラフを更新", key="graph_manual_refresh", use_container_width=True):
    st.rerun()

  graph_data = []
  for c in st.session_state.cows:
    p = c.get("実際落札額", 0)
    w = c.get("体重", 0)
    # 0円や未入力の牛はグラフからも自動除外
    if p and p > 0 and w and w > 0:
      kp = int(round(p * 1000 / w))
      graph_data.append({
          "No": int(c['No']),
          "出場番号": f"No.{c['No']}",
          "kg単価(円)": kp,
          "raw_no": c["No"]
      })

  if graph_data:
    df_g = pd.DataFrame(graph_data).sort_values("raw_no")
    
    import altair as alt
    
    avg_kp = int(round(df_g["kg単価(円)"].mean()))

    # 折れ線グラフ（Y軸のメモリ値を明確に指定）
    line_chart = alt.Chart(df_g).mark_line(point=True).encode(
        x=alt.X('出場番号:N', sort=df_g['出場番号'].tolist(), title='出場番号'),
        y=alt.Y(
            'kg単価(円):Q', 
            scale=alt.Scale(domain=[1000, 3500], zero=False), 
            axis=alt.Axis(
                values=[1000, 1500, 2000, 2500, 3000, 3500],
                tickCount=6,
                grid=True
            ), 
            title='kg単価 (円)'
        )
    )
    
    # 平均値の横線（破線）
    rule_chart = alt.Chart(pd.DataFrame({'y': [avg_kp]})).mark_rule(
        color='#059669',
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(
        y='y:Q'
    )
    
    # グラフと平均線を重ねる
    chart = alt.layer(line_chart, rule_chart).properties(
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    min_kp = df_g["kg単価(円)"].min()
    max_kp = df_g["kg単価(円)"].max()
    avg_kp = int(df_g["kg単価(円)"].mean())
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("最低 kg単価", f"{min_kp:,} 円")
    col_stat2.metric("平均 kg単価", f"{avg_kp:,} 円")
    col_stat3.metric("最高 kg単価", f"{max_kp:,} 円")
  else:
    st.info("まだ落札額と体重が入力されたデータがありません。")


# =========================================================
# 画面5: セリ結果一覧表示画面
# =========================================================
with tab5:
  top_col1, top_col2 = st.columns([3, 1])
  with top_col1:
    st.subheader("📋 セリ結果一覧表示画面")
  with top_col2:
    if st.button(
        "🔄 今すぐ更新", key="results_manual_refresh", use_container_width=True
    ):
      st.rerun()

  my_cows = [c for c in st.session_state.cows if c.get("自社落札", False)]
  st.markdown("#### 🏆 本日落札した牛一覧")
  if my_cows:
    df_my = pd.DataFrame(my_cows)[[
        "No",
        "個体識別番号",
        "性別",
        "生年月日",
        "日齢",
        "産次",
        "体重",
        "父",
        "母の父",
        "母の祖父",
        "母の母の祖父",
        "摘要",
        "実際落札額",
        "落札者番号",
    ]]
    df_my.columns = [
        "出場番号",
        "個体識別番号",
        "性別",
        "生年月日",
        "日齢",
        "産次",
        "当日体重(kg)",
        "父牛",
        "母の父",
        "母の祖父",
        "母の母の祖父",
        "摘要",
        "落札額(千円)",
        "落札者番号",
    ]
    st.dataframe(df_my, use_container_width=True, hide_index=True)
  else:
    st.info("自社落札した牛はまだありません。")

  st.divider()

  st.markdown("#### 📑 本日のセリ結果一覧（全頭）")
  all_rows = []
  for c in st.session_state.cows:
    m = calculate_cow_metrics(c)
    all_rows.append({
        "出場番号": c["No"],
        "個体識別番号": c.get("個体識別番号", "") or "-",
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
        "損益分岐点(千円)": m["ボーダー価格"],
        "落札額(千円)": c["実際落札額"],
        "落札者番号": c.get("落札者番号", ""),
        "購入結果": (
            "自社落札"
            if c.get("自社落札", False)
            else ("他社落札" if c["実際落札額"] > 0 else "-")
        ),
    })
  df_all = pd.DataFrame(all_rows)
  st.dataframe(df_all, use_container_width=True, hide_index=True)

  st.divider()

  if st.session_state.get("kintone_sent_success"):
    st.success("✅ kintoneにデータを正常に送信しました！")
    st.info("💡 入力値をリセットし、次回のセリの準備をします。")
    if st.button(
        "🧹 画面をリセットして次のセリへ",
        type="primary",
        use_container_width=True,
    ):
      clear_backup()
      clean_cows = []
      for c in st.session_state.cows:
        new_c = c.copy()
        new_c["体重"] = 0
        new_c["実際落札額"] = 0
        new_c["落札者番号"] = ""
        new_c["自社落札"] = False
        new_c["マイナス要素"] = []
        clean_cows.append(new_c)

      next_reset_ver = st.session_state.get("reset_ver", 0) + 1
      st.session_state.reset_ver = next_reset_ver
      st.session_state.cows = clean_cows
      st.session_state.curr_idx_w = 0
      st.session_state.curr_idx_p = 0
      st.session_state.input_buffer_w = ""
      st.session_state.input_buffer_p = ""
      st.session_state.p_mode = "price"
      st.session_state.metrics_dirty = True
      st.session_state.kintone_sent_success = False
      st.rerun()
  else:
    if st.button(
        "☁️ タップでkintoneに送る", type="primary", use_container_width=True
    ):
      with st.spinner("kintoneにデータを送信中..."):
        success, msg = send_to_kintone(st.session_state.cows)
        if success:
          st.session_state.kintone_sent_success = True
          st.rerun()
        else:
          st.error(msg)