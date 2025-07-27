import streamlit as st
import os
import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import altair as alt
from huggingface_hub import hf_hub_download

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY in your .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=gemini_api_key)

MODEL_DIR = './poem_emotion_app_models'
MAX_LENGTH = 128

SUB_EMOTION_COLUMNS = [
    'Romance', 'Devotion', 'Tenderness', 'Longing', 'Affection',
    'Grief', 'Loneliness', 'Melancholy', 'Despair', 'Regret',
    'Euphoria', 'Contentment', 'Hope', 'Celebration', 'Peace',
    'Rage', 'Betrayal', 'Resentment', 'Frustration', 'Defiance',
    'Dread', 'Anxiety', 'Uncertainty', 'Vulnerability', 'Anguish'
]

MAIN_TO_SUB = {
    'Love': ['Romance', 'Devotion', 'Tenderness', 'Longing', 'Affection'],
    'Sadness': ['Grief', 'Loneliness', 'Melancholy', 'Despair', 'Regret'],
    'Joy': ['Euphoria', 'Contentment', 'Hope', 'Celebration', 'Peace'],
    'Anger': ['Rage', 'Betrayal', 'Resentment', 'Frustration', 'Defiance'],
    'Fear': ['Dread', 'Anxiety', 'Uncertainty', 'Vulnerability', 'Anguish']
}

@st.cache_resource
def load_models():
    try:
        repo_id = "salyaprh/analyze-your-poem"

        tokenizer = DistilBertTokenizer.from_pretrained(
            repo_id,
            subfolder="poem_emotion_app_models/main_emotion_tokenizer"
        )

        main_model = DistilBertForSequenceClassification.from_pretrained(
            repo_id,
            subfolder="poem_emotion_app_models/main_emotion_model"
        )

        sub_model = DistilBertForSequenceClassification.from_pretrained(
            repo_id,
            subfolder="poem_emotion_app_models/sub_emotion_model"
        )
        sub_model.eval()

        encoder_path = hf_hub_download(
            repo_id=repo_id,
            filename="poem_emotion_app_models/main_emotion_label_encoder.joblib"
        )
        main_encoder = joblib.load(encoder_path)

        gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

        return tokenizer, main_model, main_encoder, sub_model, gemini_model

    except Exception as e:
        st.error("❌ Failed to load models. Check Hugging Face repo and subfolder paths.")
        st.exception(e)
        return None, None, None, None, None


tokenizer, main_model, main_encoder, sub_model, gemini_model = load_models()

if not all([tokenizer, main_model, main_encoder, sub_model, gemini_model]):
    st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

def predict_emotions(poem_text, tokenizer, main_model, main_encoder, sub_model, main_to_sub, sub_emotion_columns):
    main_inputs = tokenizer(poem_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        main_outputs = main_model(**main_inputs)
    main_probs = torch.nn.functional.softmax(main_outputs.logits, dim=-1)
    main_pred_prob, main_pred_class = torch.max(main_probs, dim=1)
    main_emotion = main_encoder.inverse_transform([main_pred_class.item()])[0]
    main_confidence = main_pred_prob.item()

    sub_inputs = tokenizer(poem_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        sub_outputs = sub_model(**sub_inputs)
    sub_probs = torch.sigmoid(sub_outputs.logits).squeeze().numpy()

    relevant_sub_emotions = main_to_sub[main_emotion]
    relevant_sub_probs = {
        emotion: float(sub_probs[sub_emotion_columns.index(emotion)])
        for emotion in relevant_sub_emotions
    }

    sorted_sub_emotions = sorted(relevant_sub_probs.items(), key=lambda item: item[1], reverse=True)

    display_sub_emotions = {}
    if len(sorted_sub_emotions) >= 2:
        display_sub_emotions[sorted_sub_emotions[0][0]] = 0.5
        display_sub_emotions[sorted_sub_emotions[1][0]] = 0.5
        for emotion, prob in sorted_sub_emotions[2:]:
            display_sub_emotions[emotion] = 0.0
    elif len(sorted_sub_emotions) == 1:
        display_sub_emotions[sorted_sub_emotions[0][0]] = 1.0

    return {
        "main_emotion": main_emotion,
        "main_confidence": float(main_confidence),
        "sub_emotions": display_sub_emotions,
        "all_main_emotions": {
            emotion: float(prob)
            for emotion, prob in zip(main_encoder.classes_, main_probs.numpy()[0])
        }
    }

def generate_poem_interpretation(poem_text, main_emotion, sub_emotions, gemini_model):
    top_sub_emotions = sorted(sub_emotions.items(), key=lambda x: x[1], reverse=True)[:2]
    primary = top_sub_emotions[0][0] if top_sub_emotions else "unspecified emotion"
    secondary = top_sub_emotions[1][0] if len(top_sub_emotions) > 1 else None

    prompt = f"""
Analyze the following poem and provide a two-sentence interpretation focusing on the identified emotions.

Poem:
\"\"\"
{poem_text}
\"\"\"

Main Emotion: {main_emotion}
Key Sub-emotions: {primary}{f", {secondary}" if secondary else ""}

Provide a concise, two-sentence interpretation that highlights these emotions.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate interpretation: {e}"


st.title("📝 Analyze Your Poem")
st.write("Enter a poem below and click **Analyze** to get the predicted main emotion, sub-emotions, and a generated interpretation.")

poem_input = st.text_area("Enter your poem here:", height=200)

if st.button("Analyze"):
    if poem_input:
        with st.spinner("Analyzing emotions and generating interpretation..."):
            results = predict_emotions(
                poem_input,
                tokenizer,
                main_model,
                main_encoder,
                sub_model,
                MAIN_TO_SUB,
                SUB_EMOTION_COLUMNS
            )

            interpretation = generate_poem_interpretation(
                poem_input,
                results['main_emotion'],
                results['sub_emotions'],
                gemini_model
            )

            st.session_state.history.append({
                "poem": poem_input,
                "main_emotion": results['main_emotion'],
                "main_confidence": results['main_confidence'],
                "sub_emotions": results['sub_emotions'],
                "interpretation": interpretation
})


        st.subheader("📊 Analysis Results:")
        st.write(f"**Main Emotion:** {results['main_emotion']} (Confidence: {results['main_confidence']*100:.1f}%)")

        st.write("**📊 Sub-Emotion Distribution:**")
        if results['sub_emotions']:
            sub_df = pd.DataFrame({
                'Sub Emotion': list(results['sub_emotions'].keys()),
                'Probability (%)': [v * 100 for v in results['sub_emotions'].values()]
            }).sort_values(by='Probability (%)', ascending=True)  # for horizontal bars

            bar_chart = alt.Chart(sub_df).mark_bar(color='#1f77b4').encode(
                x=alt.X('Probability (%):Q', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Sub Emotion:N', sort='-x'),
                tooltip=['Sub Emotion', 'Probability (%)']
            ).properties(
                width=500,
                height=200
            )

            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.write("No specific sub-emotions identified for the predicted main emotion.")

        st.write("**🧠 Poem Interpretation:**")
        st.write(interpretation)

        with st.expander("View all main emotion probabilities"):
            st.write("**All Main Emotion Probabilities:**")
            for emotion, prob in results['all_main_emotions'].items():
                st.write(f"  - {emotion}: {prob*100:.1f}%")
    else:
        st.warning("⚠️ Please enter a poem to analyze.")

    if st.session_state.history:
        st.markdown("---")
        st.subheader("🔎 Session History")

        for idx, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Entry #{len(st.session_state.history) - idx + 1}**")
            st.write(f"> {entry['poem']}")
            st.write(f"- **Main Emotion:** {entry['main_emotion']} ({entry['main_confidence']*100:.1f}%)")
        
            if entry['sub_emotions']:
                sub_df = pd.DataFrame({
                    'Sub Emotion': list(entry['sub_emotions'].keys()),
                    'Probability (%)': [v * 100 for v in entry['sub_emotions'].values()]
                }).sort_values(by='Probability (%)', ascending=True)
            
                st.altair_chart(
                    alt.Chart(sub_df).mark_bar(color='#4c78a8').encode(
                        x=alt.X('Probability (%):Q', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('Sub Emotion:N', sort='-x'),
                        tooltip=['Sub Emotion', 'Probability (%)']
                    ).properties(height=150),
                    use_container_width=True
                )
            else:
                st.write("_No sub emotions identified._")
        
        st.write(f"**Interpretation:** {entry['interpretation']}")
        st.markdown("---")

st.sidebar.subheader("📘 About")
st.sidebar.info(
    "This app analyzes the emotion of a poem using a fine-tuned DistilBERT model "
    "and generates a short interpretation using the Gemini API."
)

st.sidebar.subheader("🛠 How to Use")
st.sidebar.markdown(
    "1. Enter your poem in the text area.\n"
    "2. Click the 'Analyze' button.\n"
    "3. View the predicted main emotion, sub-emotion distribution, and interpretation."
)