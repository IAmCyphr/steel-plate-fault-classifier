from pathlib import Path

import streamlit as st

TOKENS = {
    "surface": "#fcfcfb",
    "surface-raised": "#ffffff",
    "surface-sunken": "#f3f3f1",
    "hairline": "#e4e3df",
    "ink": "#0b0b0b",
    "ink-secondary": "#52514e",
    "ink-muted": "#74736e",
    "accent": "#256abf",
    "accent-soft": "#cde2fb",
    "accent-strong": "#184f95",
}

BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BAR_BASE = "#86b6ef"
BAR_FOCUS = "#1c5cab"
BAR_SINGLE = "#2a78d6"

CSS_FILE = Path(__file__).parent / "assets" / "style.css"


def apply():
    root = "".join(f"--{name}:{value};" for name, value in TOKENS.items())
    css = f":root{{{root}}}\n" + CSS_FILE.read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
