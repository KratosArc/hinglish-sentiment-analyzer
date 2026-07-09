"""
Streamlit demo for Hinglish Sentiment Analyzer
Run locally: streamlit run api/streamlit_demo.py
"""

import streamlit as st
from transformers import pipeline
import pandas as pd

st.set_page_config(
    page_title="🇮🇳 Hinglish Sentiment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
    background:linear-gradient(90deg,#00C9FF,#92FE9D);
    color:black;
}

textarea{
    border-radius:12px;
}

.block-container{
    padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style='text-align:center;
color:white;
padding:20px;
border-radius:15px;
background:linear-gradient(90deg,#ff512f,#dd2476);'>
🇮🇳 Hinglish Sentiment Analyzer
</h1>

<h4 style='text-align:center;color:gray'>
Analyze sentiment of code-mixed Hindi-English text using XLM-RoBERTa
</h4>
""", unsafe_allow_html=True)


# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="Jiveshwar/hinglish-sentiment-analyzer", # adjust path if needed
        tokenizer="Jiveshwar/hinglish-sentiment-analyzer"
    )

classifier = load_model()

# Input
user_text = st.text_area(
    "📝 Enter Hinglish Text",
    placeholder="Example: yeh movie bohot acchi thi yaar ❤️",
    height=180
)

if st.button("Analyze Sentiment", use_container_width=True):
    if user_text.strip():
        result = classifier(user_text)
        label = result[0]["label"]
        score = result[0]["score"]
        
        # Map labels
        label_map = {"LABEL_0": "😢 Negative", "LABEL_1": "😐 Neutral", "LABEL_2": "😊 Positive"}
        display_label = label_map.get(label, label)
        
        # Display result
        st.success(f"**Sentiment:** {display_label}")
        st.info(f"**Confidence:** {score:.2%}")
        
        # Show confidence for all classes
        st.markdown("### Confidence Scores")
        st.metric("Negative", f"{score:.2%}" if label == "LABEL_0" else "—")
        st.metric("Neutral", f"{score:.2%}" if label == "LABEL_1" else "—")
        st.metric("Positive", f"{score:.2%}" if label == "LABEL_2" else "—")
    else:
        st.warning("Please enter some text!")

with st.sidebar:

    st.title("🤖 About")

    st.success("Hinglish Sentiment Analyzer")

    st.write("Model : XLM-RoBERTa")

    st.write("Dataset : SemEval-2020 Task 9")

    st.write("Framework : Streamlit")

    st.write("Developer : Jiveshwar Singh Rathore")
st.markdown("📊 [GitHub](https://github.com/KratosArc/hinglish-sentiment-analyzer) | 🤗 [Model](https://huggingface.co/Jiveshwar/hinglish-sentiment-analyzer)")
