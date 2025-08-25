import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfminer.high_level
from langdetect import detect, DetectorFactory
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Ensure consistent language detection results
DetectorFactory.seed = 0

# ==============================================================================
# 0. EXAMPLE DATA
# ==============================================================================

EXAMPLE_CV = """
John Doe
Data Analyst | Business Intelligence

Summary
Detail-oriented Data Analyst with 3 years of experience in interpreting and analyzing data to drive successful business solutions. Proficient in data visualization and reporting. Eager to support business goals with data-driven insights.

Experience
Data Analyst, Tech Corp - Anytown, USA (2021-Present)
- Developed interactive dashboards using Tableau to track key performance indicators.
- Wrote complex SQL queries to extract and manipulate data from large relational databases.
- Communicated findings to stakeholders through clear and concise reports.

Skills
- Data Visualization: Tableau, Power BI
- Databases: SQL, PostgreSQL
- Languages: Python
- Reporting & Communication
"""

EXAMPLE_JD = """
Job Title: Senior Data Analyst

We are seeking a Senior Data Analyst to join our dynamic team. The ideal candidate will be an expert in data analysis, data visualization, and have a strong background in business intelligence. You will be responsible for creating insightful reports and dashboards.

Responsibilities:
- Analyze large datasets to identify trends and patterns.
- Build and maintain dashboards in Tableau or a similar tool.
- Use SQL to perform complex data extraction and analysis.
- Collaborate with cross-functional teams to understand data needs.
- Present data-driven recommendations to leadership.

Required Skills:
- 5+ years of experience in data analysis.
- Expertise in SQL and Python for data manipulation.
- Advanced proficiency with BI tools, especially Tableau.
- Strong knowledge of business intelligence principles.
- Excellent communication skills and teamwork.
"""

# ==============================================================================
# 1. CORE NLP & PARSING FUNCTIONS
# ==============================================================================

@st.cache_data
def parse_pdf(file):
    """Extract text from an uploaded PDF file."""
    try:
        return pdfminer.high_level.extract_text(file)
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return ""

@st.cache_resource
def get_spacy_model(language_code):
    """Download and load the appropriate SpaCy model."""
    model_map = {'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'de': 'de_core_news_sm', 'es': 'es_core_news_sm'}
    model_name = model_map.get(language_code, 'en_core_web_sm')
    if language_code not in model_map:
        st.warning(f"Language '{language_code}' not fully supported. Falling back to English.")
    try:
        return spacy.load(model_name)
    except OSError:
        with st.spinner(f"Downloading SpaCy model for language: {language_code}..."):
            spacy.cli.download(model_name)
        return spacy.load(model_name)

def extract_keywords_from_jd(jd_text, nlp):
    """UPGRADED: Extracts key skills using noun chunks and POS tagging for better quality."""
    doc = nlp(jd_text)
    keywords = set()
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower().strip())
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop and len(token.text) > 2:
            keywords.add(token.lemma_.lower().strip())
    generic_terms = ['experience', 'responsibilities', 'skills', 'requirements', 'team', 'work', 'candidate']
    keywords = [kw for kw in keywords if kw not in generic_terms]
    if not keywords: return []
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    try:
        vectorizer.fit_transform(keywords)
        return vectorizer.get_feature_names_out()
    except Exception:
        return list(keywords)[:20]

def analyze_cv(cv_text, jd_keywords):
    """Matches JD keywords against the CV and calculates the score."""
    cv_text_lower = cv_text.lower()
    matched_keywords = [kw for kw in jd_keywords if re.search(r'\b' + re.escape(kw) + r'\b', cv_text_lower)]
    missing_keywords = [kw for kw in jd_keywords if kw not in matched_keywords]
    score = (len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords.size > 0 else 0
    return score, matched_keywords, missing_keywords

def generate_suggestions(missing_keywords):
    """UPGRADED: Generates more actionable suggestions."""
    if not missing_keywords:
        return "✅ **Excellent!** Your CV appears to have great keyword coverage for this job description."
    suggestions = "### 💡 Tailoring Suggestions\n To better align your CV with the job description, consider incorporating the following keywords. Here are some ideas:\n"
    suggestions += "<ul>"
    for kw in missing_keywords:
        suggestions += f"<li>For **'{kw.title()}'**, you could describe a project or a responsibility where you demonstrated this skill. For example, add a bullet point like *'Leveraged {kw} to achieve X result.'*</li>"
    suggestions += "</ul>"
    return suggestions

def create_word_cloud(keywords, title):
    """Generates a word cloud image from a list of keywords."""
    if len(keywords) == 0: return None
    wc_text = ' '.join(keywords)
    wordcloud = WordCloud(width=400, height=200, background_color='white', collocations=False).generate(wc_text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# **** THIS IS THE FUNCTION THAT WAS NOT BEING CALLED BEFORE ****
def highlight_cv(cv_text, matched_keywords, missing_keywords):
    """Highlights matched keywords in the CV text."""
    highlighted_text = cv_text
    for keyword in matched_keywords:
        # Use regex to find whole words, case-insensitive
        highlighted_text = re.sub(
            r'\b(' + re.escape(keyword) + r')\b',
            r'<span style="background-color: #90EE90; border-radius: 4px;">\1</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text

# ==============================================================================
# 2. STREAMLIT WEB GUI
# ==============================================================================

st.set_page_config(page_title="AI CV Auto-Tailor", page_icon="📄", layout="wide")
st.title("📄 AI-Powered CV Auto-Tailor")
st.markdown("Instantly analyze your CV against a job description to improve your chances with ATS systems.")

if 'cv_text' not in st.session_state: st.session_state.cv_text = ""
if 'jd_text' not in st.session_state: st.session_state.jd_text = ""

def load_example():
    st.session_state.cv_text = EXAMPLE_CV
    st.session_state.jd_text = EXAMPLE_JD
def clear_text():
    st.session_state.cv_text = ""
    st.session_state.jd_text = ""
    st.success("Inputs cleared!")

col1, col2 = st.columns(2)
with col1:
    st.header("Your CV / Resume")
    uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.session_state.cv_text = parse_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else uploaded_file.getvalue().decode("utf-8")
    st.text_area("CV Content (auto-filled from upload)", value=st.session_state.cv_text, height=300, key="cv_text_area")
    st.session_state.cv_text = st.session_state.cv_text_area
with col2:
    st.header("The Job Description")
    st.text_area("Paste the job description here", value=st.session_state.jd_text, height=300, key="jd_text_area")
    st.session_state.jd_text = st.session_state.jd_text_area

st.markdown("---")
button_col1, button_col2, button_col3, _ = st.columns([2,1,1,3])
with button_col1: analyze_button = st.button("Tailor My CV", type="primary", use_container_width=True)
with button_col2: st.button("Load Example", on_click=load_example, use_container_width=True)
with button_col3: st.button("Clear All", on_click=clear_text, use_container_width=True)

if analyze_button:
    if not st.session_state.cv_text or not st.session_state.jd_text:
        st.warning("Please make sure both the CV and Job Description fields are filled.")
    else:
        with st.spinner("AI is analyzing your documents... This may take a moment."):
            try:
                lang = detect(st.session_state.jd_text)
                nlp = get_spacy_model(lang)
                jd_keywords = extract_keywords_from_jd(st.session_state.jd_text, nlp)
                score, matched, missing = analyze_cv(st.session_state.cv_text, jd_keywords)
                
                # **** THIS IS THE FIX: CALL THE FUNCTION AND CREATE THE VARIABLE ****
                highlighted_cv = highlight_cv(st.session_state.cv_text, matched, missing)
                
                jd_cloud = create_word_cloud(jd_keywords, "Top Keywords in Job Description")
                matched_cloud = create_word_cloud(matched, "Keywords Matched in Your CV")

                tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💡 Tailoring Suggestions", "📄 Highlighted CV"])
                with tab1:
                    st.header("Analysis Dashboard")
                    st.metric("Keyword Match Score", f"{score:.1f}%")
                    st.progress(int(score))
                    st.markdown("---")
                    cloud_col1, cloud_col2 = st.columns(2)
                    with cloud_col1:
                        if jd_cloud: st.image(jd_cloud)
                    with cloud_col2:
                        if matched_cloud: st.image(matched_cloud)
                    st.markdown("---")
                    list_col1, list_col2 = st.columns(2)
                    with list_col1:
                        st.subheader("✅ Matched Keywords")
                        st.success(", ".join(sorted(matched)) if matched else "No keywords matched.")
                    with list_col2:
                        st.subheader("❌ Missing Keywords")
                        st.warning(", ".join(sorted(missing)) if missing else "No missing keywords found!")
                with tab2:
                    st.markdown(generate_suggestions(missing), unsafe_allow_html=True)
                with tab3:
                    st.subheader("Highlighted CV Text")
                    # **** NOW THIS LINE WILL WORK CORRECTLY ****
                    st.markdown(f'<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 400px; overflow-y: scroll;">{highlighted_cv}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")

st.sidebar.header("How It Works")
st.sidebar.info(
    "1. **Language Detection:** Identifies the JD's language.\n"
    "2. **Smarter Keyword Extraction:** Uses NLP (Noun Chunks & POS tagging) to find the most important skills.\n"
    "3. **CV Matching:** Scans your CV for these keywords to find matches and gaps.\n"
    "4. **Scoring & Suggestions:** Calculates a match score and provides actionable tips to help you beat Applicant Tracking Systems (ATS)."
)