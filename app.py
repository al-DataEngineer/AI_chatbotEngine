import csv
import html
import json
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

KB_PATH = "knowledge_base.csv"
AI_AVATAR = "assets/ai_avatar.svg" if os.path.exists("assets/ai_avatar.svg") else "🤖"
USER_AVATAR = "assets/user_avatar.svg" if os.path.exists("assets/user_avatar.svg") else "🧑‍💻"

st.set_page_config(
    page_title="MinAIEngine - AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Ultra-Smooth 4K Glassmorphism CSS ---
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
    --bg-deep: #060B19;
    --bg-slate: #0F172A;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --accent-cyan: #06B6D4;
    --accent-indigo: #6366F1;
    --glass-bg: rgba(255, 255, 255, 0.04);
    --glass-bg-hover: rgba(255, 255, 255, 0.075);
    --glass-border: rgba(255, 255, 255, 0.12);
    --ease-smooth: cubic-bezier(0.16, 1, 0.3, 1);
}

/* Crisp 4K / Retina Typography & Fixed Deep Ambient Mesh Gradient */
html, body, .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

.stApp {
    background-color: var(--bg-deep) !important;
    background-image:
        radial-gradient(circle at 15% 18%, rgba(6, 182, 212, 0.16) 0%, transparent 42%),
        radial-gradient(circle at 85% 22%, rgba(99, 102, 241, 0.16) 0%, transparent 45%),
        radial-gradient(circle at 50% 88%, rgba(6, 182, 212, 0.09) 0%, transparent 50%),
        linear-gradient(180deg, #060B19 0%, #0F172A 100%) !important;
    background-attachment: fixed !important;
}

code, pre, .stCode, .stText {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide Streamlit Clutter while Preserving Sidebar Toggle */
#MainMenu, footer, [data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.25s var(--ease-smooth) !important;
}

[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(6, 182, 212, 0.18) !important;
    border-color: rgba(6, 182, 212, 0.45) !important;
}

/* Main Container Spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 7.5rem !important;
    max-width: 860px !important;
}

/* Glassmorphism Sidebar — Luxury Dark Edition */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8, 13, 28, 0.88) 0%, rgba(12, 19, 38, 0.84) 100%) !important;
    backdrop-filter: blur(26px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(26px) saturate(180%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 12px 0 40px rgba(0, 0, 0, 0.45) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

[data-testid="stSidebar"] .stMarkdown label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: #CBD5E1 !important;
}

/* Luxury Sidebar Header */
.sidebar-brand {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0 0.9rem 0;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.sidebar-brand-title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-brand-tag {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.3);
    color: #67E8F9;
}

/* Frosted Glass Input in Sidebar */
[data-testid="stSidebar"] [data-baseweb="input"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.11) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transition: border-color 0.25s var(--ease-smooth), box-shadow 0.25s var(--ease-smooth) !important;
}

[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
    border-color: rgba(6, 182, 212, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.18) !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] input {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-size: 0.85rem !important;
}

/* Override Streamlit Red Slider with Cyan-Indigo Glow */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #F8FAFC !important;
    border: 2px solid #06B6D4 !important;
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.75) !important;
    width: 16px !important;
    height: 16px !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: rgba(255, 255, 255, 0.12) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #06B6D4 0%, #6366F1 100%) !important;
}

[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #67E8F9 !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}

/* Toggle Switch: biarkan teks transparan, hanya beri gaya bersih pada label */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    background: transparent !important;
    padding: 0.25rem 0 !important;
}

/* Sidebar Clear Button Centered & Sleek */
[data-testid="stSidebar"] .stButton > button {
    text-align: center !important;
    justify-content: center !important;
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.11) !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 0.9rem !important;
    margin-top: 0.25rem !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(244, 63, 94, 0.12) !important;
    border-color: rgba(244, 63, 94, 0.45) !important;
    color: #FDA4AF !important;
    box-shadow: 0 8px 24px rgba(244, 63, 94, 0.15) !important;
}

/* Luxury Spec Card in Sidebar */
.sidebar-glass-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.045) 0%, rgba(255, 255, 255, 0.015) 100%);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin: 0.85rem 0;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.spec-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.6rem;
}

.spec-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.38rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.spec-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.spec-key {
    color: #94A3B8;
    font-weight: 500;
}

.spec-val {
    color: #E2E8F0;
    font-weight: 600;
    font-size: 0.76rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.12rem 0.48rem;
    border-radius: 6px;
}

/* Minimalist Centered Hero (No Box) with Smooth Lift-Up Transition */
.hero-minimal {
    text-align: center;
    padding: 3rem 0 1.5rem 0;
    margin-bottom: 0.75rem;
    animation: fadeSlideUp 0.55s var(--ease-smooth) both;
    transition: padding 0.55s var(--ease-smooth), transform 0.55s var(--ease-smooth);
}

.hero-minimal.is-compact {
    padding: 0.1rem 0 0.85rem 0;
    margin-bottom: 0.5rem;
    animation: heroLiftUp 0.55s var(--ease-smooth) both;
}

.hero-title {
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #F8FAFC 0%, #38BDF8 52%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    transition: font-size 0.5s var(--ease-smooth);
}

.hero-minimal.is-compact .hero-title {
    font-size: 1.65rem;
}

.hero-subtitle {
    margin: 0.55rem auto 0 auto;
    font-size: 0.96rem;
    color: var(--text-secondary);
    font-weight: 500;
    transition: font-size 0.5s var(--ease-smooth);
}

.hero-minimal.is-compact .hero-subtitle {
    font-size: 0.82rem;
    margin-top: 0.3rem;
}

@keyframes heroLiftUp {
    from {
        opacity: 0.72;
        transform: translateY(34px) scale(1.06);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.48rem;
    padding: 0.36rem 0.85rem;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.32);
    color: #A7F3D0;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

.status-badge.waiting {
    background: rgba(245, 158, 11, 0.10);
    border-color: rgba(245, 158, 11, 0.32);
    color: #FDE68A;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulseDot 2s infinite;
}

.status-badge.waiting .pulse-dot {
    background: #F59E0B;
    box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
}

/* Glass Buttons & Quick Starter Pills */
.stButton > button {
    width: 100%;
    background: rgba(255, 255, 255, 0.045) !important;
    backdrop-filter: blur(16px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.13) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.72rem 1rem !important;
    text-align: left !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22) !important;
    transition: transform 0.28s var(--ease-smooth), background 0.28s var(--ease-smooth), border-color 0.28s var(--ease-smooth), box-shadow 0.28s var(--ease-smooth) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    background: rgba(6, 182, 212, 0.12) !important;
    border-color: rgba(6, 182, 212, 0.48) !important;
    box-shadow: 0 12px 28px rgba(6, 182, 212, 0.18) !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.99) !important;
}

.starter-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin: 0.25rem 0 0.65rem 0.15rem;
}

/* Chat Message Glass Bubbles — Clean Left & Right Airy Layout */
[data-testid="stChatMessage"] {
    display: flex !important;
    align-items: flex-start !important;
    gap: 0.85rem !important;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(150%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 1rem 1.3rem !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.16) !important;
    animation: fadeSlideUp 0.38s var(--ease-smooth) both;
}

/* Pesan Pengguna di Kanan */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    width: fit-content !important;
    max-width: 72% !important;
    border-radius: 20px 20px 6px 20px !important;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.14) 0%, rgba(99, 102, 241, 0.12) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    padding: 0.85rem 1.15rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    text-align: left !important;
}

/* Pesan Asisten di Kiri */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-right: auto !important;
    margin-left: 0 !important;
    max-width: 84% !important;
    border-radius: 20px 20px 20px 6px !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #E2E8F0 !important;
    line-height: 1.7 !important;
    font-size: 0.94rem !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    border-radius: 50% !important;
    overflow: hidden !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.28) !important;
    background: transparent !important;
}

/* Expander & Reference Glass Cards */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(14px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(150%) !important;
    border: 1px solid rgba(255, 255, 255, 0.11) !important;
    border-radius: 14px !important;
    margin-top: 0.75rem !important;
    overflow: hidden !important;
    transition: border-color 0.25s var(--ease-smooth) !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(6, 182, 212, 0.35) !important;
}

.ref-glass-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.ref-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.45rem;
    border: 1px solid transparent;
}

.badge-statistika { background: rgba(6, 182, 212, 0.14); color: #67E8F9; border-color: rgba(6, 182, 212, 0.35); }
.badge-analyst { background: rgba(16, 185, 129, 0.14); color: #6EE7B7; border-color: rgba(16, 185, 129, 0.35); }
.badge-scientist { background: rgba(99, 102, 241, 0.16); color: #A5B4FC; border-color: rgba(99, 102, 241, 0.38); }
.badge-engineer { background: rgba(245, 158, 11, 0.14); color: #FCD34D; border-color: rgba(245, 158, 11, 0.35); }
.badge-ai { background: rgba(236, 72, 153, 0.14); color: #F9A8D4; border-color: rgba(236, 72, 153, 0.35); }
.badge-default { background: rgba(148, 163, 184, 0.14); color: #CBD5E1; border-color: rgba(148, 163, 184, 0.32); }

.ref-content {
    font-size: 0.84rem;
    color: var(--text-secondary);
    white-space: pre-wrap;
    line-height: 1.5;
}

/* Floating Glass Chat Input */
[data-testid="stBottom"] {
    background: linear-gradient(180deg, rgba(6, 11, 25, 0) 0%, rgba(6, 11, 25, 0.82) 38%, rgba(6, 11, 25, 0.96) 100%) !important;
    padding-top: 0.75rem !important;
    padding-bottom: 1rem !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
}

[data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.68) !important;
    backdrop-filter: blur(20px) saturate(170%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(170%) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45) !important;
    transition: border-color 0.25s var(--ease-smooth), box-shadow 0.25s var(--ease-smooth) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(6, 182, 212, 0.65) !important;
    box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.22), 0 14px 40px rgba(6, 182, 212, 0.18) !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-secondary) !important;
}

/* Animations */
@keyframes fadeSlideUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.65); }
    70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.25); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(6, 182, 212, 0.45); }
</style>
""",
    unsafe_allow_html=True,
)

# --- Interactive 4K Water Ripple & Liquid Cursor FX ---
components.html(
    """
    <script>
    (function () {
        if (!window.matchMedia("(pointer: fine)").matches) return;
        let doc, win;
        try {
            win = window.parent;
            doc = win.document;
        } catch (e) {
            win = window;
            doc = document;
        }
        if (!doc || doc.getElementById("mina-cursor-layer")) return;

        const style = doc.createElement("style");
        style.id = "mina-cursor-style";
        style.textContent = `
            #mina-cursor-layer {
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 999999;
                overflow: hidden;
            }
            #mina-water-canvas {
                position: fixed;
                inset: 0;
                width: 100vw;
                height: 100vh;
                pointer-events: none;
            }
        `;
        doc.head.appendChild(style);

        const layer = doc.createElement("div");
        layer.id = "mina-cursor-layer";
        const canvas = doc.createElement("canvas");
        canvas.id = "mina-water-canvas";
        layer.appendChild(canvas);
        doc.body.appendChild(layer);

        const ctx = canvas.getContext("2d");
        let dpr = Math.min(win.devicePixelRatio || 1, 2.5);
        let width = win.innerWidth;
        let height = win.innerHeight;

        function resize() {
            dpr = Math.min(win.devicePixelRatio || 1, 2.5);
            width = win.innerWidth;
            height = win.innerHeight;
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        resize();
        win.addEventListener("resize", resize, { passive: true });

        const ripples = [];
        let targetX = -500, targetY = -500;
        let auraX = -500, auraY = -500;
        let lastRippleX = -500, lastRippleY = -500;
        let active = false;

        function spawnRipple(x, y, maxRadius, speed, alpha, lineWidth) {
            if (ripples.length > 45) ripples.shift();
            ripples.push({
                x,
                y,
                r: 3,
                maxR: maxRadius,
                speed,
                alpha,
                lineWidth,
            });
        }

        doc.addEventListener("mousemove", (e) => {
            targetX = e.clientX;
            targetY = e.clientY;
            if (!active) {
                auraX = targetX;
                auraY = targetY;
                active = true;
            }

            const dx = targetX - lastRippleX;
            const dy = targetY - lastRippleY;
            const dist = Math.hypot(dx, dy);

            if (dist >= 22) {
                lastRippleX = targetX;
                lastRippleY = targetY;
                spawnRipple(targetX, targetY, 48 + Math.random() * 18, 1.35, 0.42, 1.4);
            }
        }, { passive: true });

        doc.addEventListener("mousedown", (e) => {
            // Tetesan air jatuh ke genangan: 3 riak konsentris berurutan
            spawnRipple(e.clientX, e.clientY, 95, 2.1, 0.65, 1.8);
            setTimeout(() => spawnRipple(e.clientX, e.clientY, 72, 1.75, 0.45, 1.3), 75);
            setTimeout(() => spawnRipple(e.clientX, e.clientY, 50, 1.45, 0.30, 1.0), 150);
        }, { passive: true });

        doc.addEventListener("mouseleave", () => {
            active = false;
        }, { passive: true });

        function render() {
            ctx.clearRect(0, 0, width, height);

            if (active) {
                auraX += (targetX - auraX) * 0.14;
                auraY += (targetY - auraY) * 0.14;

                // Pantulan cahaya lembut di permukaan air
                const grad = ctx.createRadialGradient(auraX, auraY, 0, auraX, auraY, 130);
                grad.addColorStop(0, "rgba(56, 189, 248, 0.14)");
                grad.addColorStop(0.45, "rgba(6, 182, 212, 0.06)");
                grad.addColorStop(1, "rgba(6, 182, 212, 0)");
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(auraX, auraY, 130, 0, Math.PI * 2);
                ctx.fill();
            }

            for (let i = ripples.length - 1; i >= 0; i--) {
                const rp = ripples[i];
                rp.r += rp.speed * (1 - (rp.r / rp.maxR) * 0.45);
                const progress = rp.r / rp.maxR;
                const currentAlpha = rp.alpha * Math.pow(1 - progress, 1.5);

                if (progress >= 1 || currentAlpha <= 0.01) {
                    ripples.splice(i, 1);
                    continue;
                }

                // Cincin riak luar (cyan terang)
                ctx.beginPath();
                ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(103, 232, 249, ${currentAlpha.toFixed(3)})`;
                ctx.lineWidth = rp.lineWidth * (1 - progress * 0.5);
                ctx.stroke();

                // Cincin pantulan dalam (indigo/biru lembut)
                if (rp.r > 10) {
                    ctx.beginPath();
                    ctx.arc(rp.x, rp.y, rp.r * 0.65, 0, Math.PI * 2);
                    ctx.strokeStyle = `rgba(129, 140, 248, ${(currentAlpha * 0.45).toFixed(3)})`;
                    ctx.lineWidth = Math.max(0.6, rp.lineWidth * 0.65);
                    ctx.stroke();
                }
            }

            win.requestAnimationFrame(render);
        }

        win.requestAnimationFrame(render);
    })();
    </script>
    """,
    height=0,
)


def badge_class_for(kategori: str) -> str:
    k = kategori.lower()
    if "statistika" in k:
        return "badge-statistika"
    if "analyst" in k:
        return "badge-analyst"
    if "scientist" in k:
        return "badge-scientist"
    if "engineer" in k and "ai" not in k:
        return "badge-engineer"
    if "ai" in k:
        return "badge-ai"
    return "badge-default"


def render_hero(is_compact: bool = False, container=None):
    cls = "hero-minimal is-compact" if is_compact else "hero-minimal"
    markup = f"""
        <div class="{cls}">
            <h1 class="hero-title">🤖 MinAIEngine</h1>
            <p class="hero-subtitle">
                Asisten AI berbasis Retrieval-Augmented Generation (RAG) • Dibangun oleh Ahmad Alimin
            </p>
        </div>
    """
    target = container if container is not None else st
    target.markdown(markup, unsafe_allow_html=True)


# --- Manajemen GROQ_API_KEY (st.secrets dengan fallback input manual) ---
secret_api_key = ""
try:
    secret_api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    secret_api_key = ""

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-brand-title">Control Center</span>
            <span class="sidebar-brand-tag">RAG v2</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if secret_api_key:
        st.markdown(
            '<div class="sidebar-glass-card"><div class="spec-row"><span class="spec-key">API Key</span><span class="spec-val">Secrets Aktif</span></div></div>',
            unsafe_allow_html=True,
        )
        manual_override = st.text_input(
            "Ganti Groq API Key (Opsional)",
            type="password",
            placeholder="Kosongkan untuk pakai Secrets",
        )
        groq_api_key = manual_override.strip() or secret_api_key
    else:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
        ).strip()

    k_docs = st.slider("Dokumen Rujukan (Top-K)", min_value=1, max_value=6, value=3)
    auto_save_kb = st.toggle("Auto-Save ke Knowledge Base", value=True)

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya **MinAIEngine**. Ada yang bisa saya bantu hari ini?"}
        ]
        st.session_state.pop("pending_prompt", None)
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-glass-card">
            <div class="spec-title">Arsitektur Sistem</div>
            <div class="spec-row"><span class="spec-key">LLM Engine</span><span class="spec-val">Groq Auto</span></div>
            <div class="spec-row"><span class="spec-key">Embedding</span><span class="spec-val">MiniLM-L12</span></div>
            <div class="spec-row"><span class="spec-key">Vector DB</span><span class="spec-val">In-Memory</span></div>
            <div class="spec-row"><span class="spec-key">Framework</span><span class="spec-val">LangChain</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not groq_api_key:
    render_hero(is_compact=False)
    st.stop()


@st.cache_resource
def init_vector_store():
    df = pd.read_csv(KB_PATH)
    docs = []
    for _, row in df.iterrows():
        konten = f"Kategori: {row['kategori']}\nPertanyaan: {row['pertanyaan']}\nJawaban: {row['jawaban']}"
        docs.append(Document(page_content=konten, metadata={"kategori": row["kategori"]}))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return InMemoryVectorStore.from_documents(docs, embeddings)


vector_store = init_vector_store()


def simpan_ke_knowledge_base(pertanyaan: str, jawaban: str, kategori: str = "Percakapan") -> bool:
    """Simpan Q&A baru ke knowledge_base.csv jika belum ada, lalu tambahkan ke vector_store."""
    q_clean = pertanyaan.strip()
    a_clean = jawaban.strip()
    if not q_clean or not a_clean:
        return False

    if os.path.exists(KB_PATH):
        df = pd.read_csv(KB_PATH)
        sudah_ada = df["pertanyaan"].astype(str).str.strip().str.lower().eq(q_clean.lower()).any()
        if sudah_ada:
            return False

    butuh_newline = False
    if os.path.exists(KB_PATH) and os.path.getsize(KB_PATH) > 0:
        with open(KB_PATH, "rb") as f:
            f.seek(-1, os.SEEK_END)
            butuh_newline = f.read(1) not in (b"\n", b"\r")

    with open(KB_PATH, mode="a", encoding="utf-8", newline="") as f:
        if butuh_newline:
            f.write("\n")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([kategori, q_clean, a_clean])

    konten = f"Kategori: {kategori}\nPertanyaan: {q_clean}\nJawaban: {a_clean}"
    vector_store.add_documents([Document(page_content=konten, metadata={"kategori": kategori})])
    return True


# --- Pilih model chat Groq yang stabil dan aktif ---
client = Groq(api_key=groq_api_key)
semua_model = [m.id for m in client.models.list().data]

prioritas_model = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "gemma2-9b-it",
]

model_chat_tersedia = [
    m
    for m in semua_model
    if not any(x in m.lower() for x in ["guard", "whisper", "tts", "vision", "prompt", "safeguard"])
]

kandidat_model = [m for m in prioritas_model if m in model_chat_tersedia]
for m in model_chat_tersedia:
    if m not in kandidat_model and ("llama" in m or "gemma" in m or "qwen" in m or "mixtral" in m):
        kandidat_model.append(m)

if not kandidat_model:
    kandidat_model = model_chat_tersedia or semua_model

model_aktif = kandidat_model[0]

with st.sidebar:
    st.caption(f"⚡ Model aktif: `{model_aktif}`")

hero_slot = st.empty()


def buat_llm(nama_model: str) -> ChatGroq:
    return ChatGroq(
        api_key=groq_api_key,
        model=nama_model,
        temperature=0.3,
        max_tokens=800,
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Kamu adalah MinAIEngine, asisten AI cerdas dan serba bisa yang dikembangkan oleh Ahmad Alimin, dengan keahlian khusus di bidang Statistika, Data Analyst, Data Scientist, Data Engineer, dan AI Engineering.

Panduan menjawab:
1. Gunakan "Konteks Basis Pengetahuan" di bawah ini sebagai referensi utama apabila relevan dengan pertanyaan pengguna (terutama untuk identitas MinAIEngine dan konsep inti).
2. Jika pertanyaan pengguna tidak tercantum di dalam konteks atau membutuhkan penjelasan lebih luas (seperti contoh kode, studi kasus, atau topik di luar basis pengetahuan), JANGAN menolak—tetap jawab secara akurat, lengkap, dan membantu menggunakan pengetahuanmu sendiri.
3. Perhatikan riwayat percakapan agar jawabanmu nyambung dengan pertanyaan sebelumnya.
4. Jawab dalam bahasa Indonesia yang jelas, terstruktur, dan mudah dipahami. Jika memang tidak tahu suatu fakta spesifik atau data real-time, sampaikan dengan jujur.

Konteks Basis Pengetahuan:
{context}

Riwayat Percakapan Terakhir:
{history}""",
        ),
        ("human", "{question}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya **MinAIEngine**. Ada yang bisa saya bantu hari ini?"}
    ]

# --- Tangani Input Manual atau Quick Starter ---
typed_query = st.chat_input("Ketik pertanyaanmu untuk MinAIEngine...")
user_query = typed_query or st.session_state.pop("pending_prompt", None)
chat_started = len(st.session_state.messages) > 1 or bool(user_query)

render_hero(is_compact=chat_started, container=hero_slot)

# --- 4 Quick Starter Prompts saat riwayat chat masih kosong ---
QUICK_STARTERS = [
    "Perbedaan Data Analyst, Data Scientist, dan Data Engineer",
    "Kapan memakai uji ANOVA dibanding uji-t?",
    "Perbedaan ETL vs ELT dan Data Warehouse",
    "Cara mencegah overfitting pada Machine Learning",
]

if not chat_started:
    # --- Interactive Tailwind CSS v4 Knowledge Explorer ---
    try:
        df_kb = pd.read_csv(KB_PATH)
        domain_list = ["Statistika", "Data Analyst", "Data Scientist", "Data Engineer", "AI Engineering"]
        domain_payload = {}
        for dom in domain_list:
            sub = df_kb[df_kb["kategori"] == dom]
            domain_payload[dom] = {
                "count": int(len(sub)),
                "samples": sub["pertanyaan"].head(3).tolist(),
            }
        payload_json = json.dumps(domain_payload, ensure_ascii=False)
        total_kb = int(len(df_kb))
    except Exception:
        payload_json = "{}"
        total_kb = 0

    components.html(
        f"""
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8" />
            <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
            <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    padding: 4px;
                    background: transparent;
                    font-family: 'Plus Jakarta Sans', sans-serif;
                    color: #F8FAFC;
                    -webkit-font-smoothing: antialiased;
                }}
            </style>
        </head>
        <body>
            <div id="glass-explorer"
                 class="relative overflow-hidden rounded-2xl border border-white/15 bg-white/[0.04] p-4 backdrop-blur-xl shadow-2xl transition-all duration-300">
                <div id="spotlight"
                     class="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300"
                     style="background: radial-gradient(360px circle at 0px 0px, rgba(6,182,212,0.16), transparent 60%);"></div>

                <div class="relative z-10 flex flex-wrap items-center justify-between gap-2 mb-3">
                    <div class="flex items-center gap-2">
                        <span class="inline-flex h-2 w-2 rounded-full bg-cyan-400 animate-pulse"></span>
                        <span class="text-xs font-semibold uppercase tracking-wider text-slate-300">
                            Interactive Knowledge Explorer
                        </span>
                    </div>
                    <span class="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-cyan-300">
                        Total {total_kb} Basis Pengetahuan
                    </span>
                </div>

                <div id="tabs" class="relative z-10 flex flex-wrap gap-1.5 mb-3"></div>

                <div id="panel" class="relative z-10 space-y-1.5"></div>

                <div id="toast"
                     class="pointer-events-none fixed bottom-3 right-4 rounded-lg border border-emerald-400/40 bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-200 opacity-0 backdrop-blur-md transition-all duration-300">
                    ✓ Pertanyaan disalin!
                </div>
            </div>

            <script>
                const data = {payload_json};
                const domains = Object.keys(data);
                let active = domains[0] || "";

                const card = document.getElementById("glass-explorer");
                const spotlight = document.getElementById("spotlight");
                card.addEventListener("mousemove", (e) => {{
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    spotlight.style.opacity = "1";
                    spotlight.style.background = `radial-gradient(360px circle at ${{x}}px ${{y}}px, rgba(6,182,212,0.18), transparent 60%)`;
                }});
                card.addEventListener("mouseleave", () => {{ spotlight.style.opacity = "0"; }});

                function showToast() {{
                    const t = document.getElementById("toast");
                    t.classList.remove("opacity-0");
                    t.classList.add("opacity-100");
                    setTimeout(() => {{
                        t.classList.remove("opacity-100");
                        t.classList.add("opacity-0");
                    }}, 1500);
                }}

                function render() {{
                    const tabsEl = document.getElementById("tabs");
                    const panelEl = document.getElementById("panel");
                    tabsEl.innerHTML = "";
                    panelEl.innerHTML = "";

                    domains.forEach((d) => {{
                        const isActive = d === active;
                        const btn = document.createElement("button");
                        btn.textContent = `${{d}} (${{data[d].count}})`;
                        btn.className = isActive
                            ? "cursor-pointer rounded-xl border border-cyan-400/60 bg-cyan-500/20 px-3 py-1.5 text-xs font-semibold text-cyan-200 shadow-lg shadow-cyan-500/10 transition-all duration-200"
                            : "cursor-pointer rounded-xl border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-slate-300 hover:border-white/25 hover:bg-white/[0.08] transition-all duration-200";
                        btn.onclick = () => {{ active = d; render(); }};
                        tabsEl.appendChild(btn);
                    }});

                    if (active && data[active]) {{
                        data[active].samples.forEach((q) => {{
                            const item = document.createElement("button");
                            item.className = "group flex w-full cursor-pointer items-center justify-between rounded-xl border border-white/10 bg-slate-900/50 px-3 py-2 text-left text-xs text-slate-200 transition-all duration-200 hover:border-cyan-400/45 hover:bg-cyan-500/10";
                            const qSpan = document.createElement("span");
                            qSpan.className = "truncate pr-2";
                            qSpan.textContent = q;
                            const copySpan = document.createElement("span");
                            copySpan.className = "shrink-0 text-[11px] text-cyan-300 opacity-70 group-hover:opacity-100";
                            copySpan.textContent = "Salin ↗";
                            item.appendChild(qSpan);
                            item.appendChild(copySpan);
                            item.onclick = async () => {{
                                try {{ await navigator.clipboard.writeText(q); showToast(); }} catch (e) {{}}
                            }};
                            panelEl.appendChild(item);
                        }});
                    }}
                }}
                render();
            </script>
        </body>
        </html>
        """,
        height=245,
    )

    st.markdown('<div class="starter-label">✨ Mulai Cepat dengan Pertanyaan Populer</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for idx, starter_text in enumerate(QUICK_STARTERS):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(f"💬 {starter_text}", key=f"starter_{idx}", use_container_width=True):
                st.session_state.pending_prompt = starter_text
                st.rerun()

# --- Tampilkan Riwayat Chat ---
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else AI_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_query:
    history_text = "\n".join(
        f"{'Pengguna' if m['role'] == 'user' else 'MinAIEngine'}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("MinAIEngine sedang berpikir..."):
            retrieved_docs = vector_store.similarity_search(user_query, k=k_docs)
            context_text = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])

            answer = None
            last_error = None
            for nama_model in kandidat_model:
                try:
                    chain = prompt | buat_llm(nama_model)
                    response = chain.invoke(
                        {
                            "context": context_text,
                            "history": history_text,
                            "question": user_query,
                        }
                    )
                    answer = response.content
                    break
                except Exception as e:
                    last_error = (nama_model, e)

            if answer is not None:
                st.markdown(answer)

                with st.expander("📚 Lihat Sumber Referensi yang Diambil"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        kategori = str(doc.metadata.get("kategori", "Umum"))
                        badge_cls = badge_class_for(kategori)
                        safe_kategori = html.escape(kategori)
                        safe_content = html.escape(doc.page_content)
                        st.markdown(
                            f"""
                            <div class="ref-glass-card">
                                <span class="ref-badge {badge_cls}">Sumber {i} • {safe_kategori}</span>
                                <div class="ref-content">{safe_content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if auto_save_kb:
                    if simpan_ke_knowledge_base(user_query, answer):
                        st.caption("✅ Tersimpan otomatis ke `knowledge_base.csv`")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                failed_model, err = last_error
                st.error(f"Gagal memanggil Groq API (terakhir mencoba `{failed_model}`): {err}")
