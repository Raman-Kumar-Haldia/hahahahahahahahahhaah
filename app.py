import base64
import json
import os
import time
import urllib.parse
from io import BytesIO

import httpx
import requests
import streamlit as st
from groq import Groq

# 1. Page Config
st.set_page_config(page_title="TaniAI - Smart Assistant", page_icon="🤖")

# 2. Add your Groq API Key HERE
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

HISTORY_FILE = "chat_history.json"


# --- Data Persistence Helpers ---
def load_all_data():
    """Load all accounts and their chats from local JSON storage."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_all_data(data):
    """Save all accounts and chats to local JSON storage."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def generate_chat_title(first_user_message):
    """Generate a short title for the chat using Groq."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Create a brief 3-5 word title summarizing the user message. Output ONLY the title text, nothing else.",
                },
                {"role": "user", "content": first_user_message},
            ],
            max_tokens=15,
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception:
        return (
            first_user_message[:20] + "..."
            if len(first_user_message) > 20
            else first_user_message
        )


def fetch_generated_image(prompt):
    """Refines prompt using Groq and fetches raw image bytes with fallback servers."""
    try:
        refinement = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI image prompt engineer. Convert the user's request into a short, detailed image prompt under 25 words. Do not use quotes or special characters. Output ONLY the refined prompt.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=40,
        )
        detailed_prompt = (
            refinement.choices[0]
            .message.content.strip()
            .replace('"', "")
            .replace("'", "")
        )
    except Exception:
        detailed_prompt = prompt

    seed = int(time.time())
    encoded_prompt = urllib.parse.quote(detailed_prompt)

    # Primary URL (Flux model) and Fallback URL (Standard model)
    urls = [
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true&seed={seed}",
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}",
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and len(response.content) > 1000:
                return response.content, detailed_prompt
        except Exception:
            continue

    return None, detailed_prompt


# --- Initialize Session State ---
if "all_data" not in st.session_state:
    st.session_state.all_data = load_all_data()

if not st.session_state.all_data:
    st.session_state.all_data = {"Main Account": {}}

if "current_account" not in st.session_state:
    st.session_state.current_account = list(st.session_state.all_data.keys())[0]

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None


# --- Sidebar UI ---
st.sidebar.title("👤 Accounts & Chats")

accounts_list = list(st.session_state.all_data.keys())
current_acc_index = (
    accounts_list.index(st.session_state.current_account)
    if st.session_state.current_account in accounts_list
    else 0
)

selected_account = st.sidebar.selectbox(
    "Active Account:", accounts_list, index=current_acc_index
)

if selected_account != st.session_state.current_account:
    st.session_state.current_account = selected_account
    st.session_state.active_chat_id = None
    st.rerun()

col_acc1, col_acc2 = st.sidebar.columns(2)

with col_acc1:
    with st.popover("➕ New Acc", use_container_width=True):
        new_acc_name = st.text_input("Account Name", key="new_acc_input")
        if st.button("Create", use_container_width=True):
            clean_name = new_acc_name.strip()
            if clean_name:
                if clean_name not in st.session_state.all_data:
                    st.session_state.all_data[clean_name] = {}
                    st.session_state.current_account = clean_name
                    st.session_state.active_chat_id = None
                    save_all_data(st.session_state.all_data)
                    st.success(f"Account '{clean_name}' created!")
                    st.rerun()
                else:
                    st.warning("Account already exists!")

with col_acc2:
    with st.popover("🗑️ Delete", use_container_width=True):
        st.write(f"Delete account **'{st.session_state.current_account}'**?")
        st.caption("This will permanently remove all its chats.")
        if st.button("Confirm Delete", type="primary", use_container_width=True):
            del st.session_state.all_data[st.session_state.current_account]

            if not st.session_state.all_data:
                st.session_state.all_data = {"Main Account": {}}

            st.session_state.current_account = list(
                st.session_state.all_data.keys()
            )[0]
            st.session_state.active_chat_id = None
            save_all_data(st.session_state.all_data)
            st.rerun()

st.sidebar.markdown("---")

active_acc = st.session_state.current_account
account_chats = st.session_state.all_data.get(active_acc, {})

st.sidebar.subheader("💬 Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.active_chat_id = None
    st.rerun()

for cid in reversed(list(account_chats.keys())):
    chat_data = account_chats[cid]
    if isinstance(chat_data, dict):
        chat_title = chat_data.get("title", f"Chat {cid}")
    else:
        chat_title = f"Chat {cid}"

    is_active = st.session_state.active_chat_id == cid
    button_label = f"📌 {chat_title}" if is_active else f"📝 {chat_title}"

    col1, col2 = st.sidebar.columns([0.8, 0.2])

    with col1:
        if st.button(button_label, key=f"btn_{cid}", use_container_width=True):
            st.session_state.active_chat_id = cid
            st.rerun()

    with col2:
        if st.button("🗑️", key=f"del_{cid}", use_container_width=True):
            del st.session_state.all_data[active_acc][cid]
            save_all_data(st.session_state.all_data)

            if st.session_state.active_chat_id == cid:
                st.session_state.active_chat_id = None

            st.rerun()


# --- Main App Interface ---
st.title("🤖 TaniAI")
st.caption(
    f"Powered by Groq & Pollinations | Active Profile: **{active_acc}**"
)

if (
    st.session_state.active_chat_id
    and st.session_state.active_chat_id in account_chats
):
    selected = account_chats[st.session_state.active_chat_id]
    if isinstance(selected, dict):
        current_messages = selected.get("messages", [])
    else:
        current_messages = selected
else:
    current_messages = [
        {
            "role": "system",
            "content": f"You are TaniAI, a smart and friendly AI companion chatting with {active_acc}.",
        }
    ]

# Display Messages
for msg in current_messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                st.image(msg["content"], caption=msg.get("prompt_text"))
            else:
                st.write(msg["content"])


# --- NATIVE CHAT INPUT WITH AUDIO ---
prompt = st.chat_input(
    "Ask TaniAI or record your voice...",
    accept_audio=True,
    key="user_chat_input",
)

# Process input (text or audio)
if prompt:
    user_message = ""
    is_image_request = False

    # Handle Audio Input First
    if prompt.audio:
        with st.chat_message("user"):
            st.audio(prompt.audio, format="audio/wav")
            with st.spinner("🎙️ Groq is transcribing your voice..."):
                try:
                    audio_bytes = prompt.audio.read()
                    translation = client.audio.transcriptions.create(
                        file=("recording.wav", audio_bytes),
                        model="whisper-large-v3-turbo",
                        language="en",
                    )
                    user_message = translation.text
                    st.write(f"*(Transcribed: '{user_message}')*")
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
                    st.stop()
    # Handle Text Input
    else:
        user_message = prompt.text
        with st.chat_message("user"):
            st.write(user_message)

    # Manage Active Chat State
    if (
        st.session_state.active_chat_id is None
        or st.session_state.active_chat_id not in account_chats
    ):
        new_id = str(int(time.time()))
        title = generate_chat_title(user_message)

        st.session_state.active_chat_id = new_id

        if active_acc not in st.session_state.all_data:
            st.session_state.all_data[active_acc] = {}

        st.session_state.all_data[active_acc][new_id] = {
            "title": title,
            "messages": [
                {
                    "role": "system",
                    "content": f"You are TaniAI, a smart and friendly AI companion chatting with {active_acc}.",
                }
            ],
        }
        current_messages = st.session_state.all_data[active_acc][new_id][
            "messages"
        ]

    current_messages.append({"role": "user", "content": user_message})

    # Detect Image Request
    image_keywords = [
        "generate image", "draw", "create image", "picture of",
        "show me an image", "make an image", "paint", "sketch",
        "render", "illustration of", "photo of"
    ]
    is_image_request = any(kw in user_message.lower() for kw in image_keywords)

    # Process Assistant Output
    with st.chat_message("assistant"):
        if is_image_request:
            with st.spinner("🎨 Generating your image..."):
                img_data, detailed_prompt = fetch_generated_image(user_message)

                if img_data:
                    # Convert binary bytes to Base64 string for safe JSON storage
                    b64_img = base64.b64encode(img_data).decode("utf-8")
                    img_str = f"data:image/jpeg;base64,{b64_img}"

                    st.image(img_str, caption=f"Prompt: {detailed_prompt}")
                    current_messages.append({
                        "role": "assistant",
                        "type": "image",
                        "content": img_str,
                        "prompt_text": f"Prompt: {detailed_prompt}",
                    })
                else:
                    st.error("⚠️ Image generation server timed out. Please try sending the prompt again!")
        else:
            with st.spinner("Thinking..."):
                api_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in current_messages
                    if m.get("type") != "image"
                ]
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", messages=api_messages
                )
                reply = completion.choices[0].message.content
                st.write(reply)
                current_messages.append({"role": "assistant", "content": reply})

    # Save and Refresh UI
    st.session_state.all_data[active_acc][st.session_state.active_chat_id][
        "messages"
    ] = current_messages
    save_all_data(st.session_state.all_data)
    st.rerun()
