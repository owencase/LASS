"""Reusable Streamlit components for the horizontal note workspace."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Iterable

import streamlit as st

from src.llm import ModelStatus
from src.storage import LectureRecord, LectureSegment


def inject_theme(theme_mode: str = "dark", text_scale: int = 100) -> None:
    """Apply the compact, desktop-first LASS workspace theme."""
    theme_mode = "light" if theme_mode == "light" else "dark"
    text_scale = min(120, max(90, int(text_scale)))
    palettes = {
        "dark": {
            "scheme": "dark",
            "bg": "#0b0d10",
            "panel": "#12151a",
            "panel_2": "#191d24",
            "field": "#0f1217",
            "hover": "#222832",
            "code": "#090b0f",
            "line": "rgba(226,232,240,.14)",
            "text": "#f5f7fa",
            "muted": "#a7b0bf",
            "accent": "#76a9ff",
            "accent_soft": "rgba(118,169,255,.12)",
            "violet": "#b6a7ff",
            "danger": "#ff8a98",
            "primary_text": "#07111f",
            "ready_text": "#bce8a4",
            "ready_bg": "rgba(76,129,48,.2)",
            "warning_text": "#ffd58a",
            "warning_bg": "rgba(164,103,20,.16)",
            "control_border": "#f8fafc",
            "control_mark": "#07111f",
        },
        "light": {
            "scheme": "light",
            "bg": "#eef0f3",
            "panel": "#ffffff",
            "panel_2": "#f7f8fa",
            "field": "#ffffff",
            "hover": "#eef2f7",
            "code": "#f1f5f9",
            "line": "#e2e8f0",
            "text": "#1e293b",
            "muted": "#64748b",
            "accent": "#2563eb",
            "accent_soft": "#eff6ff",
            "violet": "#7c3aed",
            "danger": "#dc2626",
            "primary_text": "#ffffff",
            "ready_text": "#15803d",
            "ready_bg": "#dcfce7",
            "warning_text": "#92400e",
            "warning_bg": "#fffbeb",
            "control_border": "#cbd5e1",
            "control_mark": "#ffffff",
        },
    }
    palette = palettes[theme_mode]
    theme_tokens = f"""
        <style>
        :root {{
            --bg: {palette['bg']};
            --panel: {palette['panel']};
            --panel-2: {palette['panel_2']};
            --field: {palette['field']};
            --hover: {palette['hover']};
            --code-bg: {palette['code']};
            --line: {palette['line']};
            --text: {palette['text']};
            --muted: {palette['muted']};
            --accent: {palette['accent']};
            --accent-soft: {palette['accent_soft']};
            --lime: {palette['accent']};
            --violet: {palette['violet']};
            --danger: {palette['danger']};
            --primary-text: {palette['primary_text']};
            --ready-text: {palette['ready_text']};
            --ready-bg: {palette['ready_bg']};
            --warning-text: {palette['warning_text']};
            --warning-bg: {palette['warning_bg']};
            --control-border: {palette['control_border']};
            --control-mark: {palette['control_mark']};
            --text-scale: {text_scale / 100:.2f};
        }}
        </style>
    """
    st.markdown(
        theme_tokens
        + f"""
        <style>
        html {{ color-scheme: {palette['scheme']}; font-size:calc(16px * var(--text-scale)); }}
        """
        + """
        html, body, [class*="css"] {
            font-family: Inter, Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        html, body, #root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            height:100%; min-height:0; overflow:hidden !important;
        }
        body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            color: var(--text);
            background-color: var(--bg);
        }
        .stApp {
            color: var(--text);
            background:var(--bg);
        }
        .stMainBlockContainer {
            width: 100%; height:100%; min-height:0; overflow:hidden;
            max-width:none;
            padding:0;
        }
        .stMainBlockContainer > [data-testid="stVerticalBlock"] {
            display:flex; flex-direction:column; height:100%; min-height:0; overflow:hidden;
        }
        .st-key-lass-global-header {
            flex:0 0 3.5rem; width:100%; min-height:3.5rem; overflow:visible;
            padding:.4rem .75rem;
            border-bottom:1px solid var(--line); background:var(--panel);
        }
        .st-key-lass-global-header > [data-testid="stLayoutWrapper"],
        .st-key-lass-global-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
            height:2.7rem; min-height:2.7rem; align-items:center;
        }
        .lass-global-brand {
            display:flex; align-items:center; gap:.65rem; height:2.7rem;
            color:var(--text); line-height:1.05;
        }
        .lass-global-brand .lass-brand-mark {
            display:grid; flex:0 0 2rem; place-items:center; width:2rem; height:2rem;
            border-radius:.5rem; color:#fff; background:var(--violet); font-size:.9rem;
        }
        .lass-global-brand strong {
            display:block; font-size:.82rem; font-weight:800; letter-spacing:.08em;
        }
        .lass-global-brand small {
            display:block; margin-top:.2rem; color:var(--muted); font-size:.56rem; letter-spacing:.08em;
        }
        .st-key-lass-header-menu button,
        .st-key-lass-header-settings button {
            width:2.15rem !important; min-height:2.15rem !important; height:2.15rem !important;
            padding:0 !important; justify-content:center !important;
        }
        .st-key-lass-layout {
            flex:0 0 calc(100vh - 3.5rem); width:100%;
            height:calc(100vh - 3.5rem) !important;
            min-height:calc(100vh - 3.5rem); overflow:hidden;
        }
        .st-key-lass-layout-toolbar {
            flex:0 0 auto; margin:-.2rem 0 .75rem; padding:.55rem .65rem .45rem;
            border-bottom:1px solid var(--line); background:transparent;
        }
        .st-key-lass-layout-toolbar [data-testid="stCaptionContainer"] {
            min-height:2.5rem; display:flex; align-items:center;
        }
        .st-key-lass-layout > [data-testid="stLayoutWrapper"],
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
            height:100%; min-height:0; overflow:hidden;
        }
        .st-key-lass-library-pane {
            display:flex; flex-direction:column; height:100%; min-height:0; overflow:hidden;
            padding:.75rem .65rem;
            border-right:1px solid var(--line); border-radius:0;
            background:var(--panel-2);
        }
        .st-key-lass-library-pane > [data-testid="stLayoutWrapper"] {
            flex:0 0 auto; min-height:0; overflow:visible;
        }
        .st-key-lass-library-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-library-content) {
            flex:1 1 0; height:100%; min-height:0; overflow:hidden;
        }
        .st-key-lass-library-content {
            display:flex; flex:1 1 0; flex-direction:column; width:100%; height:100%;
            min-height:0; overflow:hidden;
        }
        .st-key-lass-library-content > [data-testid="stElementContainer"]:last-child {
            flex:1 1 0; min-height:0; overflow:hidden;
        }
        .st-key-lass-library-content > [data-testid="stElementContainer"]:last-child > div {
            width:100%; height:100%; min-height:0;
        }
        .st-key-lass-library-content [data-testid="stCustomComponentV1"] {
            display:block !important; width:100%; height:100% !important;
            min-height:0; overflow:hidden;
        }
        .st-key-lass-library-content [data-testid="stCustomComponentV1"] iframe {
            height:100% !important; min-height:0 !important;
        }
        .st-key-lass-preferences {
            width:100%; min-height:0;
        }
        .st-key-lass-preferences [data-testid="stExpanderDetails"] {
            max-height:min(62vh, 36rem); overflow-y:auto;
            scrollbar-width:thin; scrollbar-color:var(--line) transparent;
        }
        .st-key-lass-workspace-pane {
            height:100%; min-height:0; overflow-y:auto; overflow-x:hidden;
            padding:.7rem 1rem 1rem; border:0;
            background:var(--bg); scrollbar-gutter:stable;
            scrollbar-width:thin; scrollbar-color:var(--line) transparent;
        }
        .st-key-lass-ai-pane {
            height:100%; min-height:0; overflow-y:auto; overflow-x:hidden;
            padding:.75rem;
            border-left:1px solid var(--line); border-radius:0;
            background:var(--panel); scrollbar-gutter:stable;
            scrollbar-width:thin; scrollbar-color:var(--line) transparent;
        }
        .st-key-lass-library-scroll {
            min-height:0; max-height:calc(100vh - 12.5rem); overflow:auto;
            padding-right:.15rem; scrollbar-width:thin; scrollbar-color:var(--line) transparent;
        }
        .st-key-lass-library-pane .stButton button {
            justify-content:flex-start; text-align:left;
        }
        .lass-current-note {
            display:flex; align-items:center; min-height:2.35rem; padding:0 .15rem;
            color:var(--text); font-size:.84rem; font-weight:700;
            overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
        }
        .st-key-toggle-library-panel {
            width:2.35rem !important; height:2.35rem !important; margin:0 !important;
        }
        .st-key-toggle-library-panel button {
            width:2.35rem !important; height:2.35rem !important; min-height:2.35rem !important;
            padding:0 !important; text-align:center !important; justify-content:center;
            border-color:var(--line) !important; background:var(--panel) !important;
            box-shadow:none; font-size:1rem;
        }
        .st-key-lass-document-pane { margin-top:.3rem; }
        .st-key-lass-document-pane > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-lass-memo-pane > [data-testid="stVerticalBlockBorderWrapper"] {
            padding:.85rem 1rem;
        }
        .st-key-lass-document-pane > [data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow:0 4px 14px rgba(15,23,42,.08);
        }
        .st-key-lass-document-pane h4, .st-key-lass-memo-pane h4 { margin:.05rem 0; }
        .st-key-lass-memo-pane { margin-top:.75rem; }
        .lass-document-empty {
            display:flex; min-height:clamp(13rem,42vh,22rem); flex-direction:column; align-items:center;
            justify-content:center; gap:.45rem; border:1px dashed var(--line);
            border-radius:.8rem; color:var(--muted); background:var(--field); text-align:center;
        }
        .lass-document-empty span { color:var(--accent); font-size:2rem; }
        .lass-document-empty b { color:var(--text); font-size:.82rem; }
        .lass-document-empty small { max-width:22rem; color:var(--muted); line-height:1.55; }
        .lass-live-empty {
            display:flex; min-height:56vh; flex-direction:column; align-items:center;
            justify-content:center; gap:.7rem; padding:1rem; border:1px dashed var(--line);
            border-radius:.85rem; color:var(--muted); background:var(--field); text-align:center;
        }
        .lass-live-empty span { color:var(--danger); font-size:1rem; }
        .lass-live-empty b { max-width:20rem; color:var(--text); font-size:.78rem; line-height:1.6; }
        .st-key-lass-recording-actions, .st-key-lass-summary-actions {
            position:sticky; z-index:12; bottom:0; margin-top:.55rem; padding:.45rem 0 .05rem;
            background:linear-gradient(transparent 0,var(--panel) 28%);
        }
        header[data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"],
        .stDeployButton { display:none !important; }
        #MainMenu, footer { display:none !important; }
        h1, h2, h3 { letter-spacing: -.035em; }
        .stApp p, .stApp li, .stApp label,
        [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
        [data-testid="stCaptionContainer"] { color: var(--text); }
        [data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

        .lass-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 3.7rem;
            margin-bottom: .9rem;
            padding: 0 .35rem .85rem;
            border-bottom:1px solid var(--line);
        }
        .lass-brand { display:flex; align-items:center; gap:.72rem; }
        .lass-logo {
            display:grid; place-items:center; width:2.15rem; height:2.15rem;
            border-radius:.72rem; color:var(--primary-text); background:var(--accent);
            box-shadow:0 0 28px var(--accent-soft); font-size:.76rem; font-weight:900;
        }
        .lass-brand b { display:block; font-size:.93rem; letter-spacing:.09em; }
        .lass-brand span { color:var(--muted); font-size:.72rem; }
        .lass-top-meta { display:flex; align-items:center; gap:.65rem; color:var(--muted); font-size:.74rem; }
        .lass-chip {
            display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .68rem;
            border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.025);
        }
        .lass-chip i { width:.4rem; height:.4rem; border-radius:50%; background:var(--ready-text); }

        .lass-pane-title {
            display:flex; align-items:center; justify-content:space-between; min-height:2rem;
            margin-bottom:.55rem;
        }
        .lass-pane-title b { font-size:.79rem; letter-spacing:.01em; }
        .lass-pane-title span { color:var(--muted); font-size:.58rem; letter-spacing:.08em; }
        .lass-kicker { color:var(--violet); font-size:.68rem; font-weight:800; letter-spacing:.13em; }

        .lass-library-empty, .lass-empty, .lass-info, .lass-error, .lass-warning {
            border:1px solid var(--line); border-radius:1rem; background:rgba(255,255,255,.022);
        }
        .lass-library-empty { padding:1.2rem; color:var(--muted); font-size:.76rem; line-height:1.6; }
        .lass-empty {
            min-height:66vh; display:grid; place-items:center; padding:3rem; text-align:center;
            background:radial-gradient(circle at 50% 38%, rgba(155,130,255,.11), transparent 17rem), var(--panel);
        }
        .lass-empty-mark {
            display:grid; place-items:center; width:4rem; height:4rem; margin:0 auto 1.2rem;
            border:1px solid color-mix(in srgb, var(--accent) 38%, transparent); border-radius:1.25rem;
            color:var(--accent); background:var(--accent-soft); font-size:1.3rem;
        }
        .lass-empty h2 { margin:0 0 .55rem; font-size:1.65rem; }
        .lass-empty p { max-width:30rem; margin:0; color:var(--muted); font-size:.85rem; line-height:1.7; }

        .lass-work-head {
            display:flex; justify-content:space-between; align-items:start; gap:1rem;
            padding:1.25rem 1.35rem; border:1px solid var(--line); border-radius:1.15rem;
            background:linear-gradient(145deg, var(--panel-2), var(--panel));
        }
        .lass-work-head h1 { margin:.25rem 0 .38rem; font-size:clamp(1.45rem,2.1vw,2.15rem); }
        .lass-work-meta { color:var(--muted); font-size:.72rem; }
        .lass-status {
            flex:0 0 auto; padding:.42rem .65rem; border-radius:.6rem;
            color:var(--ready-text); background:var(--ready-bg); font-size:.64rem; font-weight:850;
        }
        .lass-status.failed { color:var(--danger); background:rgba(255,126,140,.08); }
        .lass-status.processing { color:var(--violet); background:var(--accent-soft); }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color:var(--line) !important; border-radius:.75rem !important;
            color:var(--text); background:var(--panel) !important;
        }
        [data-testid="stExpander"] {
            overflow:hidden; border:1px solid var(--line) !important;
            border-radius:.8rem !important; background:var(--panel) !important;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary { color:var(--text) !important; background:var(--panel) !important; }
        [data-testid="stExpander"] summary:hover {
            color:var(--text) !important; background:var(--hover) !important;
        }
        [data-testid="stAlert"], [data-testid="stStatusWidget"] {
            color:var(--text) !important; border-color:var(--line) !important;
            background:var(--panel-2) !important;
        }
        [data-testid="stAlert"] *, [data-testid="stToast"] * {
            color:var(--text) !important;
        }
        /* Notifications and BaseWeb portals live outside .stApp. Theme their
           actual surfaces as well as their text; Streamlit's base stays dark. */
        [data-testid="stAlertContainer"], [data-testid="stToast"],
        [data-baseweb="notification"], [data-baseweb="toast"],
        [role="dialog"], [data-testid="stDialog"] [role="dialog"],
        [data-baseweb="modal"], [data-baseweb="tooltip"],
        [data-testid="stTooltipContent"] {
            color:var(--text) !important; background:var(--panel-2) !important;
            border-color:var(--line) !important; color-scheme:inherit;
        }
        [data-testid="stAlertContainer"] *, [role="dialog"] *,
        [data-baseweb="notification"] *, [data-baseweb="tooltip"] *,
        [data-testid="stTooltipContent"] * { color:var(--text) !important; }
        [data-testid="stAlertContainer"] a, [data-testid="stToast"] a {
            color:var(--accent) !important;
        }
        [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"],
        [data-testid="stToast"] [data-testid="stMarkdownContainer"] {
            background:transparent !important;
        }
        [data-baseweb="select"] *, [data-baseweb="menu"] *,
        [data-testid="stRadio"] p, [data-testid="stCheckbox"] p,
        [data-testid="stSlider"] label, [data-testid="stAudioInput"] * {
            color:var(--text) !important;
        }
        [data-baseweb="tag"] { color:var(--text) !important; background:var(--hover) !important; }
        [data-testid="stForm"], [data-testid="stPopoverBody"] {
            color:var(--text) !important; background:var(--panel) !important; border-color:var(--line) !important;
        }
        [data-testid="stPopoverBody"] {
            width:min(34rem,calc(100vw - 2rem)) !important;
            max-width:calc(100vw - 2rem) !important;
            max-height:min(78vh,48rem); overflow:auto;
        }
        [data-testid="stPopoverBody"] > div,
        [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
        [data-testid="stPopoverBody"] [data-testid="stTabs"],
        [data-baseweb="popover"] > div {
            color:var(--text) !important; background:var(--panel) !important;
        }
        [data-testid="stFormSubmitButton"] button {
            color:var(--text) !important; background:var(--panel-2) !important; border-color:var(--line) !important;
        }
        [data-testid="stFormSubmitButton"] button * { color:inherit !important; }
        :focus-visible { outline:2px solid var(--accent) !important; outline-offset:2px; }
        [data-testid="stFileUploader"] {
            padding:.25rem; border:1px solid var(--line); border-radius:.9rem;
            background:rgba(255,255,255,.018);
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height:7.4rem; border:1px dashed var(--line); border-radius:.7rem;
            background:var(--field);
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color:var(--accent); background:var(--hover) !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            color:var(--text) !important; border-color:var(--line) !important;
            background:var(--panel-2) !important;
        }
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploader"] section + div {
            color:var(--muted) !important;
        }
        [data-testid="stFileUploaderDropzone"] span:first-child {
            color:var(--text) !important;
        }
        [data-testid="stAudioInput"] {
            padding:.8rem; border:1px solid var(--line); border-radius:.9rem;
            color:var(--text) !important; background:var(--field) !important;
        }
        [data-testid="stAudioInput"] button {
            color:var(--text) !important; border-color:var(--line) !important;
            background:var(--panel-2) !important;
        }
        div[data-baseweb="select"] > div, div[data-baseweb="base-input"], textarea, input {
            color:var(--text) !important; border-color:var(--line) !important;
            background:var(--field) !important;
        }
        div[data-baseweb="select"] > div:hover, div[data-baseweb="base-input"]:hover,
        textarea:hover, input:hover {
            color:var(--text) !important; border-color:var(--accent) !important;
            background:var(--hover) !important;
        }
        input::placeholder, textarea::placeholder { color:var(--muted) !important; opacity:.82; }
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
            color:var(--text) !important; border-color:var(--line) !important;
            background:var(--panel-2) !important;
        }
        li[role="option"] { color:var(--text) !important; background:var(--panel-2) !important; }
        li[role="option"]:hover, li[aria-selected="true"] { background:var(--hover) !important; }
        [data-testid="stMarkdownContainer"] pre,
        [data-testid="stMarkdownContainer"] code {
            color:var(--text) !important; background:var(--code-bg) !important;
        }

        .stButton > button, .stDownloadButton > button,
        button[data-testid^="stBaseButton-"] {
            color:var(--text) !important;
            min-height:2.35rem; border-color:var(--line); border-radius:.5rem;
            background:var(--panel); text-align:left; transition:.14s ease;
        }
        .stButton > button *, .stDownloadButton > button *,
        button[data-testid^="stBaseButton-"] * { color:inherit !important; }
        .stButton > button:hover, .stDownloadButton > button:hover {
            color:var(--text) !important; border-color:var(--accent) !important;
            background:var(--hover) !important;
        }
        .stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            color:var(--primary-text) !important; border:0; background:var(--accent);
            font-weight:800; text-align:center;
        }
        .stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            color:var(--primary-text) !important; border:0 !important;
            background:var(--accent) !important; filter:brightness(1.08);
        }
        .stButton > button:disabled, .stDownloadButton > button:disabled,
        [data-testid="stFormSubmitButton"] button:disabled {
            color:var(--muted) !important; border-color:var(--line) !important;
            background:var(--panel-2) !important; opacity:.68;
            transform:none !important; filter:none !important;
        }
        .stButton > button:disabled:hover, .stDownloadButton > button:disabled:hover,
        [data-testid="stFormSubmitButton"] button:disabled:hover {
            color:var(--muted) !important; border-color:var(--line) !important;
            background:var(--panel-2) !important; transform:none !important;
        }
        [aria-disabled="true"] { color:var(--muted) !important; }

        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
            min-height:2rem; border-radius:.55rem; transition:background .14s ease;
        }
        [data-testid="stRadio"] label:hover, [data-testid="stCheckbox"] label:hover {
            color:var(--text) !important; background:var(--hover) !important;
        }
        [data-testid="stRadio"] label:hover *, [data-testid="stCheckbox"] label:hover * {
            color:var(--text) !important;
        }
        label[data-baseweb="radio"] > div:first-child {
            box-sizing:border-box !important; flex:0 0 1.08rem !important;
            width:1.08rem !important; height:1.08rem !important;
            border:2px solid var(--control-border) !important;
            background:var(--field) !important; opacity:1 !important;
            box-shadow:0 0 0 1px color-mix(in srgb, var(--bg) 42%, transparent);
        }
        label[data-baseweb="radio"] > div:first-child > div {
            background:transparent !important;
        }
        label[data-baseweb="radio"]:has(input:checked) > div:first-child {
            border-color:var(--control-border) !important; background:var(--accent) !important;
            box-shadow:0 0 0 2px var(--accent-soft);
        }
        label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {
            width:.42rem !important; height:.42rem !important;
            background:var(--control-mark) !important;
        }
        label[data-baseweb="checkbox"] > div:first-child {
            box-sizing:border-box !important; flex:0 0 1.08rem !important;
            width:1.08rem !important; height:1.08rem !important;
            border:2px solid var(--control-border) !important;
            background:var(--field) !important; opacity:1 !important;
            box-shadow:0 0 0 1px color-mix(in srgb, var(--bg) 42%, transparent);
        }
        [data-testid="stCheckbox"] label > span:first-child {
            box-sizing:border-box !important; flex:0 0 1.08rem !important;
            width:1.08rem !important; height:1.08rem !important;
            border:2px solid var(--control-border) !important;
            border-radius:.22rem !important; background:var(--field) !important;
            opacity:1 !important;
        }
        [data-testid="stCheckbox"] label > span:first-child > div {
            border:0 !important; background:transparent !important;
        }
        label[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
            border-color:var(--control-border) !important; background:var(--accent) !important;
        }
        [data-testid="stCheckbox"] label:has(input:checked) > span:first-child {
            border-color:var(--control-border) !important;
            background:var(--accent) !important;
        }
        label[data-baseweb="checkbox"]:has(input:checked) svg {
            color:var(--control-mark) !important; fill:var(--control-mark) !important;
        }
        [data-testid="stCheckbox"] input[type="checkbox"] {
            accent-color:var(--accent) !important;
        }

        [data-testid="stPopover"] button,
        button[data-testid="stPopoverButton"] {
            color:var(--text) !important; border:1px solid var(--line) !important;
            background:var(--field) !important;
        }
        [data-testid="stPopover"] button *,
        button[data-testid="stPopoverButton"] * { color:inherit !important; }

        [data-testid="stButtonGroup"] { width:100%; }
        [data-testid="stButtonGroup"] > div { width:100%; display:flex; }
        [data-testid="stButtonGroup"] button {
            flex:1 1 0; min-width:0; min-height:2.4rem;
            justify-content:center; color:var(--muted) !important;
            border-color:var(--line) !important; background:var(--field) !important;
            white-space:nowrap;
        }
        [data-testid="stButtonGroup"] button:hover {
            color:var(--text) !important; border-color:var(--control-border) !important;
            background:var(--hover) !important;
        }
        [data-testid="stButtonGroup"] button[aria-pressed="true"] {
            color:var(--primary-text) !important; border-color:var(--accent) !important;
            background:var(--accent) !important; font-weight:800;
            box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--control-border) 38%, transparent);
        }
        [data-testid="stButtonGroup"] button[aria-pressed="true"] * {
            color:var(--primary-text) !important;
        }

        [data-testid="stCustomComponentV1"] iframe {
            display:block; width:100%; border:0; border-radius:.5rem;
            color-scheme:inherit; background:transparent !important;
        }

        .lass-new-note-copy {
            display:flex; flex-direction:column; gap:.28rem; margin-bottom:1rem;
            padding:1rem 1.1rem; border:1px solid var(--line); border-radius:.9rem;
            background:linear-gradient(125deg, color-mix(in srgb, var(--violet) 9%, transparent), var(--accent-soft));
        }
        .lass-new-note-copy b { color:var(--text); font-size:.9rem; }
        .lass-new-note-copy span { color:var(--muted); font-size:.72rem; }
        .lass-chat-empty {
            margin:.6rem 0 1rem; padding:1.15rem; border:1px dashed var(--line);
            border-radius:.9rem; color:var(--muted); background:rgba(255,255,255,.018);
            font-size:.78rem; line-height:1.65; text-align:center;
        }
        [data-testid="stChatMessage"] {
            color:var(--text) !important; border:1px solid var(--line);
            border-radius:.9rem; background:var(--panel) !important;
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] > div {
            color:var(--text) !important; border-color:var(--line) !important;
            background:var(--field) !important;
        }

        .lass-metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; margin:.75rem 0; }
        .lass-metric { padding:.75rem .85rem; border:1px solid var(--line); border-radius:.75rem; background:rgba(255,255,255,.018); }
        .lass-metric b { display:block; font-size:.86rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .lass-metric span { color:var(--muted); font-size:.62rem; }
        .lass-info { padding:.82rem .9rem; color:var(--muted); font-size:.7rem; line-height:1.55; }
        .lass-info b { display:block; margin-bottom:.18rem; color:var(--text); }
        .lass-error { padding:1rem; color:var(--danger); font-size:.8rem; line-height:1.55; }
        .lass-warning {
            margin:.75rem 0; padding:.9rem 1rem; color:var(--warning-text);
            border-color:color-mix(in srgb, var(--warning-text) 30%, transparent);
            background:var(--warning-bg); font-size:.76rem; line-height:1.6;
        }
        .lass-warning b { display:block; margin-bottom:.2rem; color:var(--warning-text); }
        .lass-resummary-guide {
            margin:.72rem 0; padding:.72rem .85rem; border:1px solid var(--line);
            border-left:3px solid var(--accent); border-radius:.7rem;
            color:var(--muted); background:var(--accent-soft); font-size:.72rem; line-height:1.55;
        }
        .lass-pdf-counter {
            display:flex; align-items:center; justify-content:center; min-height:2.55rem;
            color:var(--muted); font-size:.74rem; font-variant-numeric:tabular-nums;
        }
        .lass-timeline {
            max-height:53vh; overflow:auto; padding:.2rem .9rem;
            border:0; border-radius:0; background:var(--panel) !important;
        }
        .lass-timeline-row {
            display:grid; grid-template-columns:4.1rem minmax(0,1fr); gap:.8rem;
            padding:.72rem .15rem; border-bottom:1px solid var(--line);
            color:var(--text); background:transparent !important;
        }
        .lass-timeline-row:last-child { border-bottom:0; }
        .lass-time { color:var(--accent); font: .67rem ui-monospace,SFMono-Regular,Menlo,monospace; }
        .lass-speech { color:var(--text); font-size:.84rem; line-height:1.67; }

        [data-baseweb="tab-list"] {
            display:flex !important;
            gap:.25rem; width:100%; padding:.25rem; border-radius:.75rem;
            background:var(--field) !important;
        }
        button[data-baseweb="tab"] {
            flex:1 1 0; justify-content:center; width:auto; min-width:0; min-height:2.45rem;
            color:var(--muted) !important; border-radius:.55rem; background:transparent !important;
            font-size:.76rem; white-space:nowrap;
        }
        button[data-baseweb="tab"]:hover {
            color:var(--text) !important; background:var(--hover) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color:var(--text) !important; background:var(--panel-2) !important;
        }
        [data-baseweb="tab-highlight"] { background:var(--accent) !important; }

        /* AI Lecture Notes reference shell */
        .stMainBlockContainer { padding:0; }
        .stMainBlockContainer > [data-testid="stVerticalBlock"] { gap:0 !important; }
        .st-key-lass-layout {
            flex:0 0 100vh; height:100vh !important; min-height:100vh;
        }
        [class*="st-key-lass-layout-resizer-"] {
            position:absolute !important; width:0 !important; height:0 !important;
            min-height:0 !important; overflow:hidden !important;
        }
        .st-key-lass-layout,
        .st-key-lass-library-pane, .st-key-lass-workspace-pane,
        .st-key-lass-ai-pane { gap:0 !important; }

        .st-key-lass-library-pane {
            padding:0; border:0; border-right:1px solid var(--line);
            border-radius:0; background:var(--panel-2);
        }
        .st-key-lass-library-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-sidebar-header) {
            flex:0 0 3.5rem; height:3.5rem; overflow:hidden;
        }
        .st-key-lass-library-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-library-body) {
            flex:1 1 0; min-height:0; overflow:hidden;
        }
        .st-key-lass-library-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-sidebar-footer) {
            flex:0 0 4rem; height:4rem; overflow:visible;
        }
        .st-key-lass-sidebar-header {
            height:3.5rem; padding:0 1rem; border-bottom:1px solid color-mix(in srgb,var(--line) 58%,transparent);
            background:var(--panel-2);
        }
        .st-key-lass-sidebar-header .lass-global-brand { height:3.5rem; }
        .st-key-lass-sidebar-header .lass-global-brand strong {
            font-size:.82rem; letter-spacing:0;
        }
        .st-key-lass-sidebar-header .lass-global-brand small {
            font-size:.58rem; letter-spacing:0;
        }
        .st-key-lass-library-body {
            display:flex; flex-direction:column; gap:.35rem !important;
            width:100%; height:100%; min-height:0; padding:.75rem;
            overflow:hidden; background:var(--panel-2);
        }
        .st-key-lass-library-body > [data-testid="stLayoutWrapper"]:has(.st-key-lass-library-content) {
            flex:1 1 0; min-height:0; overflow:hidden;
        }
        .st-key-all-notes-button button,
        .st-key-favorites-button button,
        .st-key-recent-button button {
            min-height:2.25rem; padding:0 .75rem; justify-content:flex-start !important;
            border-color:transparent !important; color:var(--text) !important;
            background:transparent !important; box-shadow:none; font-size:.78rem;
            font-weight:500;
        }
        .st-key-all-notes-button button[data-testid="stBaseButton-primary"],
        .st-key-favorites-button button[data-testid="stBaseButton-primary"],
        .st-key-recent-button button[data-testid="stBaseButton-primary"] {
            border-color:#f1f5f9 !important; background:var(--panel) !important;
            box-shadow:0 1px 3px rgba(15,23,42,.08); font-weight:600;
        }
        .st-key-all-notes-button button *,
        .st-key-favorites-button button *,
        .st-key-recent-button button * { color:var(--text) !important; }
        .st-key-lass-library-body .lass-pane-title {
            min-height:1.5rem; margin:.45rem 0 0; padding:0 .55rem;
        }
        .st-key-lass-library-body .lass-pane-title b {
            color:var(--muted); font-size:.62rem; text-transform:uppercase; letter-spacing:.1em;
        }
        .st-key-lass-library-body .lass-pane-title span { display:none; }
        .st-key-lass-library-body input { min-height:2.25rem; font-size:.74rem; }
        .st-key-lass-sidebar-footer {
            height:4rem; padding:.7rem .75rem; border-top:1px solid color-mix(in srgb,var(--line) 58%,transparent);
            background:var(--panel-2);
        }
        .st-key-lass-sidebar-footer > [data-testid="stLayoutWrapper"],
        .st-key-lass-sidebar-footer [data-testid="stHorizontalBlock"] { height:100%; align-items:center; }
        .lass-local-user { display:flex; align-items:center; gap:.55rem; }
        .lass-local-user > span {
            display:grid; place-items:center; width:2rem; height:2rem; border-radius:50%;
            color:#fff; background:linear-gradient(135deg,var(--accent),var(--violet));
            font-size:.62rem; font-weight:800;
        }
        .lass-local-user b, .lass-local-user small { display:block; }
        .lass-local-user b { font-size:.72rem; }
        .lass-local-user small { margin-top:.16rem; color:var(--muted); font-size:.58rem; }

        .st-key-lass-workspace-pane {
            display:flex; height:100%; padding:0; overflow:hidden;
            border:0; background:var(--bg);
        }
        .st-key-lass-workspace-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-center-header) {
            flex:0 0 3.5rem; height:3.5rem; overflow:hidden;
        }
        .st-key-lass-workspace-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-document-memo-layout) {
            flex:1 1 0; min-height:0; overflow:hidden;
        }
        .st-key-lass-document-memo-layout {
            display:flex; flex-direction:column; width:100%; height:100%; min-height:0;
            gap:0 !important; overflow:hidden; background:var(--panel);
        }
        .st-key-lass-document-memo-layout > [data-testid="stLayoutWrapper"]:has(.st-key-lass-document-pane) {
            flex:1 1 0; min-height:0; overflow:hidden;
        }
        .st-key-lass-document-memo-layout > [data-testid="stLayoutWrapper"]:has(.st-key-lass-memo-pane) {
            flex:0 0 31.5%; min-height:8rem; overflow:hidden;
        }
        .st-key-lass-center-header {
            height:3.5rem; padding:.4rem .75rem; border-bottom:1px solid var(--line);
            background:var(--panel);
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"],
        .st-key-lass-center-header [data-testid="stHorizontalBlock"] {
            height:100%; align-items:center; gap:.38rem !important;
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
            gap:.2rem !important;
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width:0;
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-toggle-library-panel),
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-toggle-right-panel) {
            flex:0 0 2rem !important; width:2rem !important;
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.lass-document-breadcrumb) {
            flex:1 1 10rem !important; width:auto !important;
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.lass-pdf-counter) {
            flex:0 0 6.5rem !important; width:6.5rem !important;
        }
        .st-key-lass-center-header > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.lass-pdf-zoom-value) {
            flex:0 0 6.8rem !important; width:6.8rem !important;
            border-left:1px solid var(--line);
        }
        .st-key-lass-center-header button {
            min-height:2rem; height:2rem; padding:0 .45rem; font-size:.65rem;
            white-space:nowrap; justify-content:center; border-color:transparent !important;
            background:transparent !important; box-shadow:none !important;
        }
        .st-key-toggle-library-panel button,
        .st-key-toggle-right-panel button { width:2rem !important; height:2rem !important; min-height:2rem !important; }
        .st-key-toggle-library-panel button p,
        .st-key-toggle-right-panel button p { display:none; }
        .st-key-toggle-library-panel button::before,
        .st-key-toggle-right-panel button::before {
            content:""; display:block; width:.92rem; height:.78rem;
            border:1.4px solid var(--muted); border-radius:.16rem;
            background:linear-gradient(90deg,transparent 0 29%,var(--muted) 29% 37%,transparent 37%);
        }
        .st-key-toggle-right-panel button::before { transform:rotate(180deg); }
        [class*="st-key-pdf-previous-"] button,
        [class*="st-key-pdf-next-"] button,
        [class*="st-key-pdf-zoom-out-"] button,
        [class*="st-key-pdf-zoom-in-"] button {
            width:2rem !important; min-width:2rem !important; padding:0 !important;
            color:var(--muted) !important; font-size:.9rem !important;
        }
        .lass-pdf-counter, .lass-pdf-zoom-value {
            display:flex; align-items:center; justify-content:center; height:2rem; min-height:0;
            color:var(--muted); font-size:.61rem; font-variant-numeric:tabular-nums;
            white-space:nowrap;
        }
        .lass-document-breadcrumb {
            display:flex; align-items:center; gap:.45rem; height:2rem; min-width:0;
            color:var(--text); font-size:.7rem; white-space:nowrap;
        }
        .lass-document-breadcrumb span { color:var(--muted); }
        .lass-document-breadcrumb b { overflow:hidden; text-overflow:ellipsis; }
        .lass-document-breadcrumb i { color:#cbd5e1; font-style:normal; }
        .lass-document-breadcrumb em {
            flex:0 0 auto; padding:.18rem .45rem; border:1px solid #bbf7d0;
            border-radius:999px; color:#15803d; background:#f0fdf4;
            font-size:.58rem; font-style:normal; font-weight:600;
        }
        .lass-document-breadcrumb em.processing { color:#1d4ed8; border-color:#bfdbfe; background:#eff6ff; }
        .lass-document-breadcrumb em.failed { color:#b91c1c; border-color:#fecaca; background:#fef2f2; }
        .lass-document-breadcrumb em.draft { color:var(--muted); border-color:var(--line); background:var(--panel-2); }

        .st-key-lass-document-pane {
            position:relative; display:flex; flex-direction:column; width:100%; height:100%;
            min-height:0; margin:0; padding:1.25rem; gap:0 !important;
            overflow:auto; background:var(--bg);
        }
        .st-key-lass-document-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-document-empty-viewer) {
            flex:1 1 0; width:100%; height:100%; min-height:0; overflow:hidden;
        }
        .st-key-lass-document-empty-viewer {
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            width:100%; max-width:51.25rem; height:100%; min-height:14rem; margin:0 auto;
            padding:2rem; gap:.65rem !important; overflow:auto; border-radius:.25rem;
            background:var(--panel); box-shadow:0 8px 30px rgba(0,0,0,.08);
        }
        .lass-document-empty-copy {
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            gap:.45rem; color:var(--muted); text-align:center;
        }
        .lass-document-empty-copy span { color:var(--accent); font-size:2rem; }
        .lass-document-empty-copy b { color:var(--text); font-size:.82rem; }
        .lass-document-empty-copy small { max-width:22rem; color:var(--muted); line-height:1.55; }
        .st-key-lass-document-pane .lass-empty {
            width:100%; max-width:51.25rem; min-height:0; aspect-ratio:16 / 9;
            margin:0 auto; border:0; border-radius:.125rem; background:var(--panel);
            box-shadow:0 8px 30px rgba(0,0,0,.08);
        }
        .st-key-lass-document-pane > [data-testid="stElementContainer"]:has(.lass-pdf-stage),
        .st-key-lass-document-pane > [data-testid="stLayoutWrapper"]:has(.lass-pdf-stage),
        .st-key-lass-document-pane .stMarkdown:has(.lass-pdf-stage),
        .st-key-lass-document-pane [data-testid="stMarkdownContainer"]:has(.lass-pdf-stage) {
            flex:1 1 0; width:100%; height:100%; min-height:0;
        }
        .lass-pdf-stage {
            display:flex; align-items:flex-start; justify-content:center; width:100%; height:100%;
            min-height:0; padding:.5rem; overflow:auto;
        }
        .lass-pdf-page {
            display:block; width:auto; height:auto; max-width:100%; max-height:100%;
            object-fit:contain; border-radius:.125rem; background:var(--panel);
            box-shadow:0 8px 30px rgba(0,0,0,.08);
            transform:scale(var(--lass-pdf-zoom,1)); transform-origin:top center;
        }

        .st-key-lass-memo-pane {
            height:100%; margin:0; padding:0; gap:0 !important; overflow:hidden;
            border-top:1px solid var(--line); background:var(--panel);
        }
        .st-key-lass-memo-pane > [data-testid="stLayoutWrapper"]:first-child,
        .st-key-lass-memo-pane > [data-testid="stElementContainer"]:first-child {
            flex:0 0 3rem; min-height:3rem; padding:.5rem 1.5rem .35rem;
            border-bottom:1px solid color-mix(in srgb,var(--line) 45%,transparent);
        }
        .st-key-lass-memo-pane > [data-testid="stElementContainer"]:has(textarea) {
            flex:1 1 0; min-height:0; padding:.65rem 1.5rem;
        }
        .st-key-lass-memo-pane textarea {
            min-height:6.5rem !important; height:6.5rem !important; padding:0 !important;
            border:0 !important; border-radius:0; color:var(--text) !important;
            background:transparent !important; font-size:.82rem; line-height:1.6;
            resize:none;
        }
        .st-key-lass-memo-pane [data-baseweb="base-input"] {
            border:0 !important; background:transparent !important;
        }
        .st-key-lass-memo-pane [data-testid="stTextAreaRootElement"] {
            border:0 !important; background:transparent !important;
        }
        .lass-memo-title { display:flex; align-items:center; gap:.45rem; height:2rem; }
        .lass-memo-title > span { color:var(--accent); font-size:1rem; }
        .lass-memo-title b { font-size:.68rem; letter-spacing:.05em; }
        .lass-memo-title small { margin-left:auto; color:#16a34a; font-size:.56rem; }
        .st-key-lass-memo-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-memo-actions) {
            flex:0 0 3rem; min-height:3rem;
        }
        .st-key-lass-memo-actions {
            height:3rem; padding:.4rem 1.5rem; border-top:1px solid color-mix(in srgb,var(--line) 58%,transparent);
            background:var(--panel-2);
        }
        .st-key-lass-memo-actions button { min-height:2rem; height:2rem; font-size:.62rem; justify-content:center; }
        .lass-format-tools { display:flex; align-items:center; gap:.1rem; height:2rem; }
        .lass-format-tools span {
            display:grid; place-items:center; width:2rem; height:2rem; border-radius:.5rem;
            color:var(--muted); font-size:.78rem; cursor:default;
        }
        .lass-format-tools span:hover { color:var(--text); background:#e5e7eb; }

        .st-key-lass-ai-pane {
            display:flex; height:100%; padding:0; overflow:hidden;
            border:0; border-left:1px solid var(--line); border-radius:0; background:var(--panel);
        }
        .st-key-lass-ai-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-right-header) {
            flex:0 0 3.5rem; height:3.5rem; overflow:visible;
        }
        .st-key-lass-ai-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-transcript-stream),
        .st-key-lass-ai-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-summary-stream) {
            flex:1 1 0; min-height:0; overflow:hidden;
        }
        .st-key-lass-ai-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-recording-actions),
        .st-key-lass-ai-pane > [data-testid="stLayoutWrapper"]:has(.st-key-lass-summary-actions) {
            flex:0 0 auto; min-height:0;
        }
        .st-key-lass-right-header {
            height:3.5rem; padding:0; border-bottom:1px solid color-mix(in srgb,var(--line) 58%,transparent);
            background:var(--panel);
        }
        .st-key-lass-right-header > [data-testid="stLayoutWrapper"],
        .st-key-lass-right-header [data-testid="stHorizontalBlock"] { height:100%; align-items:stretch; gap:0 !important; }
        .st-key-lass-right-header [data-testid="stColumn"] { height:100%; }
        .st-key-lass-right-header button {
            min-height:3.5rem; height:3.5rem; padding:0 .3rem; border:0 !important;
            border-bottom:2px solid transparent !important; border-radius:0 !important;
            background:var(--panel) !important; box-shadow:none !important;
            color:var(--muted) !important; font-size:.7rem; font-weight:600;
            justify-content:center; gap:.45rem;
        }
        .st-key-lass-right-header [class*="st-key-right-tab-"] button[kind] {
            background:var(--panel) !important;
        }
        .st-key-lass-right-header [class*="st-key-right-tab-"] button[data-testid="stBaseButton-primary"] {
            border-bottom-color:var(--violet) !important; color:var(--violet) !important;
            background:var(--panel) !important;
        }
        .st-key-lass-right-header button:hover { color:var(--text) !important; }
        .st-key-lass-right-header button[data-testid="stBaseButton-primary"]:hover { color:var(--violet) !important; }
        .st-key-lass-right-header button [data-testid="stIconMaterial"] { font-size:.95rem; }
        .st-key-lass-sidebar-footer [data-testid="stPopover"] button svg { display:none; }
        .st-key-lass-transcript-stream, .st-key-lass-summary-stream {
            height:100%; padding:1rem; overflow:auto; background:var(--panel);
        }
        .st-key-lass-transcript-stream .lass-live-empty {
            height:100%; min-height:0; border:0; border-radius:0; background:transparent;
            visibility:hidden;
        }
        .st-key-lass-recording-actions {
            position:static; margin:0; padding:1rem 1.5rem;
            border-top:1px solid color-mix(in srgb,var(--line) 58%,transparent);
            background:color-mix(in srgb,var(--panel-2) 55%,var(--panel));
        }
        .st-key-lass-summary-actions {
            position:static; margin:0; padding:.75rem 1rem;
            border-top:1px solid var(--line); background:var(--panel);
        }
        .st-key-lass-recording-actions [data-testid="stCustomComponentV1"] iframe {
            min-height:9.5rem !important; height:9.5rem !important; border-radius:0;
        }

        /* Sidebars float above the workspace so opening one never reflows the PDF or memo. */
        .st-key-lass-layout > [data-testid="stLayoutWrapper"],
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
            position:relative; width:100%; height:100%; min-height:0; overflow:hidden;
        }
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
            display:block !important;
        }
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            position:absolute !important; inset:0; width:100% !important; min-width:0 !important;
            height:100%; margin:0 !important;
        }
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-lass-workspace-pane) {
            z-index:1;
        }
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-lass-library-pane) {
            right:auto; z-index:60; width:clamp(17rem,22vw,20rem) !important;
            filter:drop-shadow(12px 0 18px rgba(15,23,42,.14));
        }
        .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-lass-ai-pane) {
            left:auto; z-index:55; width:clamp(21rem,29vw,27rem) !important;
            filter:drop-shadow(-12px 0 18px rgba(15,23,42,.14));
        }

        /* The only draggable divider is inside the center workspace. */
        .lass-center-resizer { position:relative; z-index:20; outline:none; touch-action:none; background:transparent; }
        .lass-center-resizer span { position:absolute; display:block; background:#d8dee8; transition:background .12s ease; }
        .lass-center-resizer:hover span, .lass-center-resizer.dragging span { background:#8b95a5; }
        .lass-horizontal-resizer { flex:0 0 5px; width:100%; height:5px; cursor:row-resize; }
        .lass-horizontal-resizer span { inset:2px 0; }
        .lass-vertical-resizer { grid-column:2; grid-row:1; width:5px; height:100%; cursor:col-resize; }
        .lass-vertical-resizer span { inset:0 2px; }
        .st-key-lass-document-memo-layout.lass-center-side-by-side {
            display:grid !important;
            grid-template-columns:minmax(15rem,var(--lass-document-width,62%)) 5px minmax(13.75rem,1fr);
            grid-template-rows:minmax(0,1fr);
        }
        .st-key-lass-document-memo-layout.lass-center-side-by-side > [data-testid="stLayoutWrapper"]:has(.st-key-lass-document-pane) {
            grid-column:1; grid-row:1; width:100% !important; height:100% !important;
        }
        .st-key-lass-document-memo-layout.lass-center-side-by-side > [data-testid="stLayoutWrapper"]:has(.st-key-lass-memo-pane) {
            grid-column:3; grid-row:1; width:100% !important; height:100% !important; min-height:0;
        }
        .st-key-lass-document-memo-layout.lass-center-side-by-side .st-key-lass-memo-pane {
            border-top:0; border-left:1px solid var(--line);
        }

        .st-key-toggle-library-panel button::before,
        .st-key-toggle-right-panel button::before { display:none !important; content:none !important; }
        .st-key-toggle-center-layout button { min-width:5rem; }
        .st-key-toggle-library-panel button p,
        .st-key-toggle-right-panel button p,
        .st-key-close-library-panel button p,
        .st-key-close-right-panel button p { display:none; }
        .st-key-close-library-panel button,
        .st-key-close-right-panel button {
            width:2rem !important; min-width:2rem !important; height:2rem !important;
            min-height:2rem !important; padding:0 !important; border:0 !important;
            background:transparent !important; box-shadow:none !important;
        }
        .st-key-lass-sidebar-header [data-testid="stHorizontalBlock"] { height:100%; align-items:center; }
        .st-key-lass-right-header [data-testid="stColumn"]:has(.st-key-close-right-panel) {
            flex:0 0 2.4rem !important; width:2.4rem !important;
        }

        /* 100% fits the whole page; other values scale from that fitted page. */
        .st-key-lass-document-actions {
            display:flex !important; position:absolute; z-index:15; top:1rem; right:1rem;
            justify-content:flex-end; width:auto; height:auto; overflow:visible;
        }
        [class*="st-key-document-upload-"] { display:block !important; }
        .st-key-lass-document-actions button {
            min-height:2rem; height:2rem; padding:0 .7rem; border-color:var(--line) !important;
            background:var(--panel) !important; box-shadow:0 1px 3px rgba(15,23,42,.1) !important;
        }
        .st-key-lass-document-empty-viewer .st-key-lass-document-actions {
            position:relative; inset:auto; justify-content:center; width:auto; height:auto;
        }
        .st-key-lass-document-empty-viewer > [data-testid="stLayoutWrapper"]:has(.st-key-lass-document-actions),
        .st-key-lass-document-empty-viewer > [data-testid="stElementContainer"]:has(.st-key-lass-document-actions) {
            flex:0 0 auto; width:auto; height:auto; overflow:visible;
        }

        .st-key-lass-memo-pane > [data-testid="stElementContainer"]:has([data-testid="stCustomComponentV1"]),
        .st-key-lass-memo-pane > [data-testid="stLayoutWrapper"]:has([data-testid="stCustomComponentV1"]) {
            flex:1 1 0; width:100%; height:100%; min-height:0; overflow:hidden;
        }
        .st-key-lass-memo-pane [data-testid="stCustomComponentV1"],
        .st-key-lass-memo-pane [data-testid="stCustomComponentV1"] iframe {
            display:block !important; width:100%; height:100% !important; min-height:0 !important;
        }
        .st-key-lass-transcript-stream .lass-timeline { max-height:none; overflow:visible; }

        @media (max-width: 760px) {
            .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-lass-library-pane),
            .st-key-lass-layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-lass-ai-pane) {
                width:min(92vw,23rem) !important;
            }
        }
        ::-webkit-scrollbar { width:6px; height:6px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { border-radius:10px; background:#cbd5e1; }
        ::-webkit-scrollbar-thumb:hover { background:#94a3b8; }

        .lass-model { display:flex; justify-content:space-between; gap:.75rem; }
        .lass-model b { font-size:.75rem; }
        .lass-model span { display:block; color:var(--muted); font-size:.63rem; }
        .lass-ready { color:var(--ready-text) !important; }
        .lass-missing { color:var(--danger) !important; }

        @media (max-width: 980px) {
            .stMainBlockContainer { padding:0; }
            .lass-top-meta > span { display:none; }
            .lass-empty { min-height:20rem; }
            .st-key-lass-layout {
                flex-basis:100vh; height:100vh !important; min-height:100vh;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(model_ready: bool, note_count: int) -> None:
    state = "LOCAL AI READY" if model_ready else "MODEL SETUP REQUIRED"
    st.markdown(
        f"""
        <div class="lass-topbar">
            <div class="lass-brand">
                <div class="lass-logo">LA</div>
                <div><b>LASS</b><span>LOCAL NOTE INTELLIGENCE WORKSPACE</span></div>
            </div>
            <div class="lass-top-meta">
                <span>{note_count}개의 노트가 로컬에 저장됨</span>
                <div class="lass-chip"><i></i>{state}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pane_title(title: str, meta: str = "") -> None:
    st.markdown(
        f'<div class="lass-pane-title"><b>{escape(title)}</b><span>{escape(meta)}</span></div>',
        unsafe_allow_html=True,
    )


def render_empty_workspace() -> None:
    st.markdown(
        """
        <section class="lass-empty">
            <div>
                <div class="lass-empty-mark">◉</div>
                <div class="lass-kicker">YOUR NOTE WORKSPACE</div>
                <h2>노트를 선택하거나 새로 추가하세요.</h2>
                <p>녹음본, AI 요약, 타임라인 녹취, 내 메모를 한 화면에서 분리해 관리할 수 있습니다.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_lecture_header(lecture: LectureRecord) -> None:
    status_label = {"draft": "작성 중", "ready": "정리 완료", "processing": "처리 중", "failed": "처리 실패"}.get(
        lecture.status, lecture.status
    )
    status_class = lecture.status if lecture.status in {"failed", "processing"} else ""
    date_label = _format_date(lecture.updated_at)
    st.markdown(
        f"""
        <section class="lass-work-head">
            <div>
                <div class="lass-kicker">NOTE WORKSPACE</div>
                <h1>{escape(lecture.title)}</h1>
                <div class="lass-work-meta">{escape(lecture.source_name or '내 노트')} · {escape(date_label)}</div>
            </div>
            <div class="lass-status {status_class}">{escape(status_label)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(lecture: LectureRecord) -> None:
    values = (
        (lecture.prompt_name or "-", "요약 형식"),
        (_format_duration(lecture.duration or 0), "녹음 길이"),
        (f"{lecture.segment_count}개", "녹취 구간"),
    )
    cards = "".join(
        f'<div class="lass-metric"><b>{escape(value)}</b><span>{escape(label)}</span></div>'
        for value, label in values
    )
    st.markdown(f'<div class="lass-metrics">{cards}</div>', unsafe_allow_html=True)


def render_timeline(segments: Iterable[LectureSegment]) -> None:
    rows = "".join(
        '<div class="lass-timeline-row">'
        f'<div class="lass-time">{_format_timestamp(segment.start)}</div>'
        f'<div class="lass-speech">{escape(segment.text)}</div>'
        "</div>"
        for segment in segments
    )
    if not rows:
        rows = '<div class="lass-info">아직 녹취가 없습니다.</div>'
    st.markdown(f'<div class="lass-timeline">{rows}</div>', unsafe_allow_html=True)


def render_model_status(status: ModelStatus, model_name: str = "EXAONE 4.0 · 1.2B") -> None:
    label = "READY" if status.ready else "MISSING"
    css_class = "lass-ready" if status.ready else "lass-missing"
    meta = status.size_label if status.ready else "모델 준비 필요"
    st.markdown(
        f"""
        <div class="lass-model">
            <div><b>{escape(model_name)}</b><span>{escape(meta)} · ON DEVICE</span></div>
            <b class="{css_class}">{label}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="lass-info"><b>{escape(title)}</b>{escape(copy)}</div>',
        unsafe_allow_html=True,
    )


def render_warning(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="lass-warning"><b>{escape(title)}</b>{escape(copy)}</div>',
        unsafe_allow_html=True,
    )


def render_error(message: str) -> None:
    st.markdown(f'<div class="lass-error">{escape(message)}</div>', unsafe_allow_html=True)


def build_markdown_export(lecture: LectureRecord) -> str:
    """Build an on-demand export without creating a second source of truth."""
    lines = [
        f"# {lecture.title}",
        "",
        f"- 요약 형식: {lecture.prompt_name or '-'}",
        f"- 감지 언어: {lecture.language or 'unknown'}",
        f"- 원본 파일: {lecture.source_name}",
        "",
        "## AI 요약",
        "",
        lecture.summary or "아직 요약이 없습니다.",
        "",
        "## 전사 · 추출 텍스트",
        "",
        lecture.transcript or "아직 전사하거나 추출한 텍스트가 없습니다.",
        "",
        "## 내 메모",
        "",
        lecture.notes or "",
        "",
    ]
    if lecture.segments:
        lines.extend(("## 타임스탬프 녹취", ""))
        lines.extend(
            f"- `{_format_timestamp(segment.start)}` {segment.text}"
            for segment in lecture.segments
        )
    return "\n".join(lines).rstrip() + "\n"


def format_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y.%m.%d %H:%M")
    except (TypeError, ValueError):
        return value
