import streamlit as st
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
from ddgs import DDGS
from newspaper import Article
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# French press sources
# -----------------------------
FRENCH_NEWS_DOMAINS = [

    # center / generalist
    "lemonde.fr",
    "francetvinfo.fr",
    "ouest-france.fr",

    # center-right
    "lefigaro.fr",
    "lesechos.fr",

    # left
    "liberation.fr",
    "nouvelobs.com",

    # right
    "valeursactuelles.com",

    # radical right / far right
    "fdesouche.com",
    "ripostelaique.com",
    "boulevardvoltaire.fr",
    "egaliteetreconciliation.fr",
    "reseauinternational.net",

    # international
    "france24.com",
    "rfi.fr"
]

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from streamlit_mic_recorder import speech_to_text
    MICRO_AVAILABLE = True
except Exception:
    speech_to_text = None
    MICRO_AVAILABLE = False


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="DOXA Detector",
    page_icon="🧠",
    layout="wide",
)

st.image("banner2.png", use_container_width=True)
st.caption("Cognitive calibration laboratory — M = (G + N) − D")
st.markdown("---")

st.markdown("""
<style>
div[data-testid="stProgressBar"] > div > div > div > div {
    height: 20px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Unique EN texts
# -----------------------------
T = {
    "settings": "Settings",
    "load_example": "Load example",
    "show_method": "Show method",
    "hard_fact_score_scale": "Hard Fact Score scale",
    "scale_0_5": "very fragile",
    "scale_6_9": "doubtful",
    "scale_10_14": "plausible but needs cross-checking",
    "scale_15_20": "structurally robust",
    "topic_section": "Analyze multiple articles by topic",
    "topic": "Topic to analyze",
    "topic_placeholder": "e.g. artificial intelligence",
    "analyze_topic": "📰 Analyze 10 articles on this topic",
    "searching": "Searching and analyzing articles...",
    "articles_analyzed": "articles analyzed.",
    "analyzed_articles": "Analyzed articles",
    "avg_hard_fact": "Average Hard Fact",
    "avg_classic_score": "Average classic score",
    "topic_doxa_index": "Topic doxa index",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "credibility_score_dispersion": "Credibility score dispersion",
    "article_label": "Article",
    "no_exploitable_articles_found": "No usable articles found for this topic.",
    "enter_keyword_first": "Enter a keyword or topic first.",
    "url": "Analyze an article by URL",
    "load_url": "🌐 Load article from URL",
    "article_loaded_from_url": "Article loaded from URL.",
    "unable_to_retrieve_text": "Unable to retrieve text from this URL.",
    "paste_url_first": "Paste a URL first.",
    "paste": "Paste an article or text here",
    "analyze": "🔍 Analyze article",
    "manual_paste": "manual copy-paste",
    "loaded_url_source": "article loaded from URL",
    "text_source": "Text source",
    "paste_text_or_load_url": "Paste a text or load a URL, then click “🔍 Analyze article”.",
    "classic_score": "Classic score",
    "improved_score": "Improved score",
    "hard_fact_score": "Hard Fact Score",
    "help_classic_score": "M = (G + N) − D",
    "help_improved_score": "Adds V and penalty R",
    "help_hard_fact_score": "Stricter checking of claims and sources",
    "credibility_gauge": "Credibility gauge",
    "fragile": "Fragile",
    "fragile_message": "The text shows strong structural or factual weaknesses.",
    "doubtful": "Doubtful",
    "doubtful_message": "The text contains some credible elements, but remains highly uncertain.",
    "plausible": "Plausible",
    "plausible_message": "The text seems broadly plausible, but still needs verification.",
    "robust": "Robust",
    "robust_message": "The text has a fairly solid structural and factual foundation.",
    "score": "Score",
    "verdict": "Verdict",
    "summary": "Analysis summary",
    "strengths_detected": "Strengths detected",
    "few_strong_signals": "Few strong signals detected.",
    "weaknesses_detected": "Weaknesses detected",
    "no_major_weakness": "No major weakness detected by the heuristic.",
    "presence_of_source_markers": "Presence of source or data markers",
    "verifiability_clues": "Verifiability clues detected: links, figures, dates, or percentages",
    "text_contains_nuances": "The text contains nuance, limitations, or counterpoints",
    "text_evokes_robust_sources": "The text refers to potentially robust or institutional sources",
    "some_claims_verifiable": "Some claims are grounded enough to be properly verified",
    "overly_assertive_language": "Overly assertive or absolutist language",
    "notable_emotional_sensational_charge": "Notable emotional or sensational charge",
    "almost_total_absence_of_verifiable_elements": "Almost total absence of verifiable elements",
    "text_too_short": "Text too short to seriously support a strong claim",
    "multiple_claims_very_fragile": "Several central claims are very fragile given the available cues",
    "hard_fact_checking_by_claim": "Claim-by-claim fact-checking",
    "claim": "Claim",
    "status": "Status",
    "verifiability": "Verifiability",
    "risk": "Risk",
    "number": "Number",
    "date": "Date",
    "named_entity": "Named entity",
    "attributed_source": "Attributed source",
    "yes": "Yes",
    "no": "No",
    "to_verify": "To verify",
    "rather_verifiable": "Rather verifiable",
    "very_fragile": "Very fragile",
    "low_credibility": "Low credibility",
    "prudent_credibility": "Prudent credibility",
    "rather_credible": "Rather credible",
    "strong_credibility": "Strong credibility",
    "paste_longer_text": "Paste a slightly longer text to get a finer mapping of claims.",
    "llm_analysis": "Mécroyance analysis for systems",
    "llm_intro": "This section applies the models derived from the treatise to evaluate a system’s cognitive posture.",
    "overconfidence": "Overconfidence (asymmetry)",
    "calibration": "Relative calibration (ratio)",
    "revisability": "Revisability (R)",
    "cognitive_closure": "Cognitive closure",
    "interpretation": "Interpretation",
    "llm_metrics": "Derived metrics",
    "zone_closure": "Cognitive closure zone: certainty exceeds cognitive grounding.",
    "zone_stability": "Revisable stability zone: mécroyance accompanies without dominating.",
    "zone_lucidity": "Growing lucidity zone: doubt structures cognition.",
    "zone_rare": "Rare zone: highly integrated and reflective cognition.",
    "zone_pansapience": "Hypothetical pan-sapience: limiting horizon of an almost fully revisable cognition.",
    "zone_asymptote": "Ideal asymptote: totality of knowledge and integration, without rigidification.",
    "out_of_spectrum": "Value outside the theoretical spectrum.",
    "external_corroboration_module": "🔎 External corroboration module",
    "external_corroboration_caption": "This module searches for external sources that may confirm, nuance, or contradict the central claims of the pasted text.",
    "corroboration_in_progress": "Searching for corroboration...",
    "generated_query": "Generated query",
    "no_strong_sources_found": "No sufficiently solid source found for this claim.",
    "no_corroboration_found": "No usable corroboration found.",
    "corroborated": "Corroborated",
    "mixed": "Mixed",
    "not_corroborated": "Not corroborated",
    "insufficiently_documented": "Insufficiently documented",
    "corroboration_verdict": "Corroboration verdict",
    "match_score": "Match score",
    "contradiction_signal": "Contradiction signal",
    "detected": "Detected",
    "not_detected": "Not detected",
    "ai_module": "AI module",
    "ai_module_caption": "The AI re-reads the heuristic analysis and formulates a more synthetic critical reading.",
    "generate_ai_analysis": "✨ Generate AI analysis",
    "ai_unavailable": "AI module unavailable: missing OpenAI key or library not installed.",
    "ai_analysis_result": "AI analysis",
    "method": "Method",
    "original_formula": "Original formula",
    "articulated_knowledge_density": "G: articulated knowledge density — sources, figures, names, references, verifiable traces.",
    "integration": "N: integration — context, nuance, reservations, argumentative coherence.",
    "assertive_rigidity": "D: assertive rigidity — unsupported certainties, rhetorical overdrive.",
    "disclaimer": "This app replaces neither a journalist, nor a researcher, nor a clerk of reality. But it already removes a few masks from the text that struts about.",
}


# -----------------------------
# 3D cognitive triangle
# -----------------------------
def plot_cognitive_triangle_3d(G: float, N: float, D: float):
    G_pt = [10, 0, 0]
    N_pt = [0, 10, 0]
    D_pt = [0, 0, 10]
    P = [G, N, D]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    verts = [[G_pt, N_pt, D_pt]]
    tri = Poly3DCollection(verts, alpha=0.18, edgecolor="black", linewidths=1.5)
    ax.add_collection3d(tri)

    ax.plot([G_pt[0], N_pt[0]], [G_pt[1], N_pt[1]], [G_pt[2], N_pt[2]], linewidth=2)
    ax.plot([N_pt[0], D_pt[0]], [N_pt[1], D_pt[1]], [N_pt[2], D_pt[2]], linewidth=2)
    ax.plot([D_pt[0], G_pt[0]], [D_pt[1], G_pt[1]], [D_pt[2], G_pt[2]], linewidth=2)

    ax.scatter(*G_pt, s=80)
    ax.scatter(*N_pt, s=80)
    ax.scatter(*D_pt, s=80)

    ax.text(G_pt[0] + 0.3, G_pt[1], G_pt[2], "G", fontsize=12, weight="bold")
    ax.text(N_pt[0], N_pt[1] + 0.3, N_pt[2], "N", fontsize=12, weight="bold")
    ax.text(D_pt[0], D_pt[1], D_pt[2] + 0.3, "D", fontsize=12, weight="bold")

    ax.scatter(*P, s=140, marker="o")
    ax.text(P[0] + 0.2, P[1] + 0.2, P[2] + 0.2, "Text", fontsize=11, weight="bold")

    ax.plot([0, G], [0, 0], [0, 0], linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, N], [0, 0], linestyle="--", linewidth=1)
    ax.plot([0, 0], [0, 0], [0, D], linestyle="--", linewidth=1)
    ax.plot([0, G], [0, N], [0, D], linestyle=":", linewidth=1.5)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.set_xlabel("G — gnōsis")
    ax.set_ylabel("N — nous")
    ax.set_zlabel("D — doxa")
    ax.set_title("3D cognitive triangle")
    ax.view_init(elev=24, azim=35)

    return fig


# -----------------------------
# OpenAI client
# -----------------------------
def get_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


client = get_openai_client()


# -----------------------------
# Header
# -----------------------------
st.title("DOXA Detector")

with st.container(border=True):
    st.subheader("Analyze the reliability of a text")
    st.write(
        "DOXA Detector helps determine whether a text rests on solid facts "
        "or on persuasive rhetoric."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Paste a text")
        st.write("Copy an article or excerpt into the analysis area.")

    with col2:
        st.markdown("### 2️⃣ Analyze")
        st.write("The app examines sources, claims, and nuance.")

    with col3:
        st.markdown("### 3️⃣ Understand")
        st.write("Get a credibility score and a claim analysis.")

    st.caption(
        "This tool does not claim whether a text is true or false: "
        "it simply helps clarify how solid the information is."
    )


# -----------------------------
# Cognition model
# -----------------------------
class Cognition:
    def __init__(self, gnosis: float, nous: float, doxa: float):
        self.G = self.clamp(gnosis)
        self.N = self.clamp(nous)
        self.D = self.clamp(doxa)
        self.M = self.compute_mecroyance()

    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
        return max(min_val, min(max_val, value))

    def compute_mecroyance(self) -> float:
        return (self.G + self.N) - self.D

    def interpret(self) -> str:
        m = self.M
        if m < 0:
            return T["zone_closure"]
        if 0 <= m <= 10:
            return T["zone_stability"]
        if 10 < m <= 17:
            return T["zone_lucidity"]
        if 17 < m < 19:
            return T["zone_rare"]
        if 19 <= m < 20:
            return T["zone_pansapience"]
        if m == 20:
            return T["zone_asymptote"]
        return T["out_of_spectrum"]


# -----------------------------
# Example
# -----------------------------
SAMPLE_ARTICLE = (
    "Artificial intelligence will replace 80% of jobs by 2030, according to a shocking study published yesterday by the firm GlobalTech. "
    "The 45-page report states that the finance and healthcare sectors will be the hardest hit. "
    "\"This is an unprecedented revolution,\" says Jean Dupont, a robotics expert. "
    "However, some economists such as Marie Curie remain cautious: \"These figures need to be qualified, because new jobs will appear.\" "
    "The study specifies that 12 million jobs could be created in Europe. "
    "Despite this, concern is growing among workers who fear for their future. "
    "It is absolutely certain that we are heading toward a major social crisis if nothing is done immediately."
)


# -----------------------------
# Helpers
# -----------------------------
def clamp(n: float, minn: float, maxn: float) -> float:
    return max(min(maxn, n), minn)


@st.cache_data(show_spinner=False, ttl=3600)
def extract_article_from_url(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=1800)
def search_articles_by_keyword(keyword: str, max_results: int = 10) -> List[Dict]:
    articles = []
    seen_urls = set()

    api_key = st.secrets.get("NEWS_API_KEY")
    from_date_iso = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    # -----------------------------
    # 1) Priority: NewsAPI
    # -----------------------------
    if api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": keyword,
            "language": "fr",
            "sortBy": "publishedAt",
            "pageSize": max_results * 3,
            "apiKey": api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                for art in data.get("articles", []):
                    article_url = art.get("url")
                    title = art.get("title", "Untitled")
                    source = art.get("source", {}).get("name", "Unknown source")
                    published_at = art.get("publishedAt", "")

                    if not article_url or article_url in seen_urls:
                        continue

                    seen_urls.add(article_url)

                    articles.append({
                        "title": title,
                        "url": article_url,
                        "source": source,
                        "published_at": published_at,
                    })

                    if len(articles) >= max_results:
                        return articles

        except Exception as e:
            st.warning(f"NewsAPI error: {e}")

    # -----------------------------
    # 2) DDGS fallback
    # -----------------------------
    try:
        with DDGS() as ddgs:
            query = f"{keyword} news France"
            results = list(ddgs.text(query, max_results=max_results * 5))

            for r in results:
                url = r.get("href", "")
                title = r.get("title", "Untitled")

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                articles.append({
                    "title": title,
                    "url": url,
                    "source": url.split("/")[2] if "://" in url else url,
                    "published_at": "",
                })

                if len(articles) >= max_results:
                    break

    except Exception as e:
        st.warning(f"DDGS error: {e}")

    return articles


# -----------------------------
# Mécroyance / lie gauge
# -----------------------------
def compute_lie_gauge(G: float, N: float, D: float, M: float):

    ME_gauge = (2 * D) - (G + N)
    delta = ME_gauge - M

    # controlled amplification
    amp = 8.0

    strength = min(abs(delta) / amp, 1.0)

    if delta <= 0:
        # mécroyance zone
        gauge = 0.5 * (1 - strength)

        if gauge > 0.35:
            label = "Moderate mécroyance"
            color = "#ca8a04"
        else:
            label = "Strong mécroyance"
            color = "#a16207"

    else:
        # lie zone
        gauge = 0.5 + (0.5 * strength)

        if gauge < 0.65:
            label = "Possible lie"
            color = "#f97316"
        elif gauge < 0.85:
            label = "Probable lie"
            color = "#dc2626"
        else:
            label = "Extreme lie"
            color = "#991b1b"

    return round(gauge, 3), label, color, round(ME_gauge, 2)


@dataclass
class Claim:
    text: str
    has_number: bool
    has_date: bool
    has_named_entity: bool
    has_source_cue: bool
    absolutism: int
    emotional_charge: int
    verifiability: float
    risk: float
    status: str


SOURCE_CUES = [
    "according to", "states", "declares", "report", "study", "expert",
    "source", "says", "writes", "published", "announces", "confirms", "reveals",
]

ABSOLUTIST_WORDS = [
    "always", "never", "absolutely", "certain", "proof",
    "proven", "undeniable", "all", "none",
]

EMOTIONAL_WORDS = [
    "shock", "incredible", "terrible", "fear", "threat",
    "scandal", "revolution", "urgent", "catastrophe", "crisis",
]

NUANCE_MARKERS = [
    "however", "yet", "nevertheless", "still", "but",
    "qualify", "caution", "possible", "perhaps", "seems",
]


def analyze_claim(sentence: str) -> Claim:
    has_number = bool(re.search(r"\d+", sentence))
    has_date = bool(
        re.search(
            r"\d{4}|january|february|march|april|may|june|july|august|september|october|november|december",
            sentence,
            re.I,
        )
    )
    has_named_entity = bool(re.search(r"[A-Z][a-z]+ [A-Z][a-z]+|[A-Z]{2,}", sentence))
    has_source_cue = any(cue in sentence.lower() for cue in SOURCE_CUES)

    absolutism = sum(1 for word in ABSOLUTIST_WORDS if word in sentence.lower())
    emotional_charge = sum(1 for word in EMOTIONAL_WORDS if word in sentence.lower())

    v_score = clamp((has_number * 5) + (has_date * 5) + (has_named_entity * 5) + (has_source_cue * 5), 0, 20)
    r_score = clamp((absolutism * 7) + (emotional_charge * 7), 0, 20)

    if v_score < 5:
        status = T["very_fragile"]
    elif v_score < 12:
        status = T["to_verify"]
    else:
        status = T["rather_verifiable"]

    return Claim(
        text=sentence,
        has_number=has_number,
        has_date=has_date,
        has_named_entity=has_named_entity,
        has_source_cue=has_source_cue,
        absolutism=absolutism,
        emotional_charge=emotional_charge,
        verifiability=v_score,
        risk=r_score,
        status=status,
    )


def analyze_article(text: str) -> Dict:
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    article_length = len(words)

    source_markers = len(re.findall(r"|".join(re.escape(c) for c in SOURCE_CUES), text.lower()))
    citation_like = len(re.findall(r'"|\'|«|»', text))
    nuance_markers = len(re.findall(r"|".join(re.escape(c) for c in NUANCE_MARKERS), text.lower()))

    G = clamp(source_markers * 1.5 + citation_like * 0.5, 0, 10)
    N = clamp(nuance_markers * 2 + (article_length / 100), 0, 10)

    certainty = len(re.findall(r"certain|absolutely|proven|evident|undeniable", text.lower()))
    emotional = len(re.findall(r"|".join(re.escape(w) for w in EMOTIONAL_WORDS), text.lower()))

    D = clamp(certainty * 2 + emotional * 1.5, 0, 10)
    M = round((G + N) - D, 1)
    V = clamp(G * 0.8 + N * 0.2, 0, 10)
    R = clamp(D * 0.7 + (emotional * 1.2), 0, 10)
    improved = round((G + N + V) - (D + R), 1)

    claims = [analyze_claim(s) for s in sentences[:15]]
    avg_claim_verifiability = sum(c.verifiability for c in claims) / len(claims) if claims else 0
    avg_claim_risk = sum(c.risk for c in claims) / len(claims) if claims else 0
    source_quality = clamp(source_markers * 3 - (emotional * 2), 0, 20)

    red_flags = []
    if D > 8:
        red_flags.append("Saturated doxa")
    if emotional > 5:
        red_flags.append("Excessive pathos")
    if G < 2:
        red_flags.append("Documentary desert")
    if article_length < 50:
        red_flags.append("Impoverished format")

    hard_fact_score_raw = (
        (0.18 * G + 0.12 * N + 0.20 * V + 0.22 * source_quality + 0.18 * avg_claim_verifiability)
        - (0.16 * D + 0.12 * R + 0.18 * avg_claim_risk + 0.9 * len(red_flags))
    )
    hard_fact_score = round(clamp(hard_fact_score_raw + 8, 0, 20), 1)

    if hard_fact_score < 6:
        verdict = T["low_credibility"]
    elif hard_fact_score < 10:
        verdict = T["prudent_credibility"]
    elif hard_fact_score < 15:
        verdict = T["rather_credible"]
    else:
        verdict = T["strong_credibility"]

    strengths = []
    if source_markers >= 2:
        strengths.append(T["presence_of_source_markers"])
    if citation_like >= 2:
        strengths.append(T["verifiability_clues"])
    if nuance_markers >= 2:
        strengths.append(T["text_contains_nuances"])
    if source_quality >= 12:
        strengths.append(T["text_evokes_robust_sources"])
    if any(c.status == T["rather_verifiable"] for c in claims):
        strengths.append(T["some_claims_verifiable"])

    weaknesses = []
    if certainty >= 3:
        weaknesses.append(T["overly_assertive_language"])
    if emotional >= 2:
        weaknesses.append(T["notable_emotional_sensational_charge"])
    if source_markers == 0 and citation_like == 0:
        weaknesses.append(T["almost_total_absence_of_verifiable_elements"])
    if article_length < 80:
        weaknesses.append(T["text_too_short"])
    weaknesses.extend(red_flags)
    if sum(1 for c in claims if c.status == T["very_fragile"]) >= 2:
        weaknesses.append(T["multiple_claims_very_fragile"])

    ME = (2 * D) - (G + N)

    return {
        "words": len(words),
        "sentences": len(sentences),
        "G": G,
        "N": N,
        "D": D,
        "M": M,
        "ME": ME,
        "V": V,
        "R": R,
        "improved": improved,
        "source_quality": source_quality,
        "avg_claim_risk": avg_claim_risk,
        "avg_claim_verifiability": avg_claim_verifiability,
        "hard_fact_score": hard_fact_score,
        "verdict": verdict,
        "profil_solidite": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "claims": claims,
        "red_flags": red_flags,
    }


@st.cache_data(show_spinner=False, ttl=1800)
def analyze_multiple_articles(keyword: str, max_results: int = 10) -> List[Dict]:
    articles = search_articles_by_keyword(keyword, max_results)
    results = []
    for art in articles:
        try:
            full_text = extract_article_from_url(art["url"])
            if len(full_text) > 120:
                analysis = analyze_article(full_text)
                results.append(
                    {
                        "Source": art["source"],
                        "Title": art["title"],
                        "Classic score": analysis["M"],
                        "Hard Fact Score": analysis["hard_fact_score"],
                        "Verdict": analysis["verdict"],
                        "URL": art["url"],
                    }
                )
        except Exception:
            continue
    return results


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_text_for_textarea(url: str) -> str:
    try:
        text = extract_article_from_url(url)
        return (text or "").strip()
    except Exception:
        return ""


# -----------------------------
# Corroboration
# -----------------------------
def extract_key_sentences_for_corroboration(text: str, max_sentences: int = 5) -> List[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 40]
    scored = []
    for s in sentences:
        score = 0
        if re.search(r"\d+", s):
            score += 2
        if re.search(r"\d{4}|january|february|march|april|may|june|july|august|september|october|november|december", s, re.I):
            score += 2
        if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}", s):
            score += 2
        if any(word in s.lower() for word in [
            "according to", "states", "declares", "report", "study",
            "expert", "source", "published", "announces", "confirms", "reveals"
        ]):
            score += 1
        if any(word in s.lower() for word in [
            "absolutely", "certain", "never", "always",
            "undeniable", "shock", "scandal", "revolution", "urgent"
        ]):
            score += 1
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for _, s in scored[:max_sentences]]


def build_search_query_from_claim(claim: str) -> str:
    claim = re.sub(r"[^\w\s%\-]", " ", claim)
    claim = re.sub(r"\s+", " ", claim).strip()
    words = claim.split()
    important_words = [w for w in words if len(w) > 3][:12]
    return " ".join(important_words)


def extract_claim_features(claim: str) -> Dict:
    numbers = re.findall(r"\d+(?:[.,]\d+)?%?", claim)
    years = re.findall(r"\b(?:19|20)\d{2}\b", claim)
    proper_names = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}", claim)
    words = re.findall(r"\b\w+\b", claim.lower())
    stopwords = {
        "the", "and", "with", "from", "that", "this", "those", "these",
        "into", "over", "under", "have", "has", "had", "were", "was",
        "are", "for", "but", "not", "you", "your", "our", "their",
        "his", "her", "its", "who", "what", "when", "then", "thus",
    }
    keywords = [w for w in words if len(w) > 4 and w not in stopwords]
    return {
        "numbers": list(set(numbers)),
        "years": list(set(years)),
        "proper_names": list(set(proper_names)),
        "keywords": list(dict.fromkeys(keywords))[:12],
    }


def score_match_between_claim_and_result(claim: str, result_text: str) -> Dict:
    features = extract_claim_features(claim)
    rt = result_text.lower()
    number_hits = sum(1 for n in features["numbers"] if n.lower() in rt)
    year_hits = sum(1 for y in features["years"] if y.lower() in rt)
    proper_name_hits = sum(1 for p in features["proper_names"] if p.lower() in rt)
    keyword_hits = sum(1 for k in features["keywords"] if k.lower() in rt)

    score = 0.0
    score += number_hits * 3
    score += year_hits * 2
    score += proper_name_hits * 3
    score += min(keyword_hits, 5) * 1.2

    contradiction_markers = [
        "false", "misleading", "incorrect", "inaccurate",
        "denied", "refuted", "no evidence",
    ]
    contradiction_signal = any(marker in rt for marker in contradiction_markers)

    return {
        "score": round(score, 1),
        "number_hits": number_hits,
        "year_hits": year_hits,
        "proper_name_hits": proper_name_hits,
        "keyword_hits": keyword_hits,
        "contradiction_signal": contradiction_signal,
    }


def classify_corroboration(matches: List[Dict]) -> str:
    if not matches:
        return "insufficient"

    best_score = max(m["match_score"]["score"] for m in matches)
    contradiction_count = sum(1 for m in matches if m["match_score"]["contradiction_signal"])
    strong_matches = sum(1 for m in matches if m["match_score"]["score"] >= 8)
    medium_matches = sum(1 for m in matches if 4 <= m["match_score"]["score"] < 8)

    if strong_matches >= 2 and contradiction_count == 0:
        return "corroborated"
    if best_score >= 8 and contradiction_count >= 1:
        return "mixed"
    if medium_matches >= 1 or best_score >= 4:
        return "mixed"
    return "not_corroborated"


def display_corroboration_verdict(code: str) -> str:
    if code == "corroborated":
        return f"🟢 {T['corroborated']}"
    if code == "mixed":
        return f"🟠 {T['mixed']}"
    if code == "not_corroborated":
        return f"🔴 {T['not_corroborated']}"
    return f"⚪ {T['insufficiently_documented']}"


def corroborate_claims(text: str, max_claims: int = 5, max_results_per_claim: int = 3) -> List[Dict]:
    claims = extract_key_sentences_for_corroboration(text, max_sentences=max_claims)
    corroboration_results = []

    trusted_domains = [
        "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "theguardian.com",
        "lemonde.fr", "lefigaro.fr", "liberation.fr", "francetvinfo.fr", "lesechos.fr",
        "who.int", "un.org", "worldbank.org", "nature.com", "science.org",
        "elpais.com", "elmundo.es", "dw.com", "spiegel.de",
    ]

    try:
        with DDGS() as ddgs:
            for claim in claims:
                query = build_search_query_from_claim(claim)
                search_results = list(ddgs.text(query, max_results=max_results_per_claim * 5))
                filtered = []
                for r in search_results:
                    url = r.get("href", "")
                    title = r.get("title", "")
                    body = r.get("body", "")
                    combined_text = f"{title} {body}"
                    if any(domain in url for domain in trusted_domains):
                        match_score = score_match_between_claim_and_result(claim, combined_text)
                        filtered.append(
                            {
                                "title": title,
                                "url": url,
                                "snippet": body,
                                "match_score": match_score,
                            }
                        )
                filtered = sorted(filtered, key=lambda x: x["match_score"]["score"], reverse=True)[:max_results_per_claim]
                verdict = classify_corroboration(filtered)
                corroboration_results.append(
                    {
                        "claim": claim,
                        "query": query,
                        "matches": filtered,
                        "verdict": verdict,
                    }
                )
    except Exception as e:
        st.warning(f"Corroboration error: {e}")

    return corroboration_results


# -----------------------------
# AI helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def generate_ai_summary(article_text: str, result: Dict, max_chars: int = 7000) -> str:
    if client is None:
        return ""

    short_text = article_text[:max_chars]
    claims_preview = []
    for c in result.get("claims", [])[:8]:
        claims_preview.append(
            {
                "claim": c.text,
                "status": c.status,
                "verifiability": c.verifiability,
                "risk": c.risk,
                "has_number": c.has_number,
                "has_date": c.has_date,
                "has_named_entity": c.has_named_entity,
                "has_source_cue": c.has_source_cue,
            }
        )

    prompt = f"""
You are a rigorous critical reading assistant.

Your task:
1. Summarize the overall credibility profile of the text.
2. Explain the difference between structural plausibility and factual robustness.
3. Identify the 3 main strengths.
4. Identify the 3 main weaknesses.
5. End with a cautious verdict.

Constraints:
- Be clear, concise, and concrete.
- Do not invent any facts.
- Do not claim with certainty that a text is true or false without strong justification.
- Rely on the heuristic metrics below.

Heuristic analysis:
{json.dumps({
    'G': result.get('G'),
    'N': result.get('N'),
    'D': result.get('D'),
    'M': result.get('M'),
    'V': result.get('V'),
    'R': result.get('R'),
    'hard_fact_score': result.get('hard_fact_score'),
    'verdict': result.get('verdict'),
    'strengths': result.get('strengths', []),
    'weaknesses': result.get('weaknesses', []),
    'claims': claims_preview,
    'red_flags': result.get('red_flags', []),
}, ensure_ascii=False, indent=2)}

Text to analyze:
{short_text}
"""

    try:
        response = client.responses.create(model="gpt-4o", input=prompt)
        return response.output_text.strip()
    except Exception as e:
        return f"AI error: {e}"


# -----------------------------
# Settings
# -----------------------------
with st.expander(T["settings"], expanded=False):
    use_sample = st.button(T["load_example"])
    show_method = st.toggle(T["show_method"], value=True)
    st.divider()
    st.subheader(T["hard_fact_score_scale"])
    st.markdown(
        f"- **0–5** : {T['scale_0_5']}\n"
        f"- **6–9** : {T['scale_6_9']}\n"
        f"- **10–14** : {T['scale_10_14']}\n"
        f"- **15–20** : {T['scale_15_20']}"
    )

if "article" not in st.session_state:
    st.session_state.article = SAMPLE_ARTICLE
if "article_source" not in st.session_state:
    st.session_state.article_source = "paste"
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_article" not in st.session_state:
    st.session_state.last_article = ""
if "multi_results" not in st.session_state:
    st.session_state.multi_results = []
if "last_keyword" not in st.session_state:
    st.session_state.last_keyword = ""

if use_sample:
    st.session_state.article = SAMPLE_ARTICLE
    st.session_state.article_source = "paste"
    st.session_state.loaded_url = ""


# -----------------------------
# Multi-article analysis
# -----------------------------
st.subheader(T["topic_section"])
keyword = st.text_input(T["topic"], placeholder=T["topic_placeholder"])

if st.button(T["analyze_topic"], key="analyze_topic"):
    if keyword.strip():
        st.info(T["searching"])
        st.session_state.multi_results = analyze_multiple_articles(keyword.strip(), max_results=10)
        st.session_state.last_keyword = keyword.strip()
    else:
        st.session_state.multi_results = []
        st.warning(T["enter_keyword_first"])

if st.session_state.get("multi_results"):
    df_multi = pd.DataFrame(st.session_state.multi_results).sort_values("Hard Fact Score", ascending=False)

    st.success(f"{len(df_multi)} {T['articles_analyzed']}")

    c1, c2 = st.columns(2)
    c1.metric(T["analyzed_articles"], len(df_multi))
    c2.metric(T["avg_hard_fact"], round(df_multi["Hard Fact Score"].mean(), 1))
    st.metric(T["avg_classic_score"], round(df_multi["Classic score"].mean(), 1))

    ecart_type_hf = df_multi["Hard Fact Score"].std()
    indice_doxa = "high" if ecart_type_hf < 1.5 else ("medium" if ecart_type_hf < 3 else "low")
    st.metric(T["topic_doxa_index"], T[indice_doxa])

    st.subheader(T["credibility_score_dispersion"])
    df_plot = df_multi.copy()
    df_plot["Article"] = [f"{T['article_label']} {i+1}" for i in range(len(df_plot))]
    st.bar_chart(df_plot.set_index("Article")["Hard Fact Score"])
    st.dataframe(df_multi, use_container_width=True, hide_index=True)

    st.markdown("### Actions on found articles")

    for i, row in df_multi.reset_index(drop=True).iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['Title']}")
            st.caption(f"{row['Source']}")

            score = row["Hard Fact Score"]
            if score <= 6:
                color, label = "🔴", "Fragile"
            elif score <= 11:
                color, label = "🟠", "Doubtful"
            elif score <= 15:
                color, label = "🟡", "Plausible"
            else:
                color, label = "🟢", "Robust"

            st.markdown(f"**{color} Credibility score: {score:.1f}/20 — {label}**")
            st.progress(score / 20)

            col1, col2 = st.columns(2)
            with col1:
                st.link_button("🌐 Open article", row["URL"], use_container_width=True)
            with col2:
                if st.button(f"📥 Load for analysis", key=f"load_article_{i}"):
                    loaded_text = fetch_text_for_textarea(row["URL"])
                    if loaded_text:
                        st.session_state.article = loaded_text
                        st.session_state.article_source = "url"
                        st.session_state.loaded_url = row["URL"]
                        st.success("Article loaded into the text area.")
                        st.rerun()
                    else:
                        st.warning("Unable to extract the text.")
elif st.session_state.get("last_keyword"):
    st.warning(T["no_exploitable_articles_found"])


# -----------------------------
# URL loading
# -----------------------------
with st.form("url_form"):
    url = st.text_input(T["url"])
    load_url_submitted = st.form_submit_button(T["load_url"])

if load_url_submitted:
    if url:
        texte = extract_article_from_url(url)
        if texte:
            st.session_state.article = texte
            st.session_state.article_source = "url"
            st.session_state.loaded_url = url
            st.success(T["article_loaded_from_url"])
            st.rerun()
        else:
            st.error(T["unable_to_retrieve_text"])
    else:
        st.warning(T["paste_url_first"])


# -----------------------------
# Analysis area
# -----------------------------
previous_article = st.session_state.article

st.markdown("### Analysis area")

with st.container(border=True):
    st.caption("Paste a text, load a URL, or dictate directly.")

    if MICRO_AVAILABLE:
        spoken_text = speech_to_text(
            language="en",
            start_prompt="🎙️ Dictate",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="speech_to_text_article"
        )

        if spoken_text:
            st.session_state.article = spoken_text
            st.session_state.article_source = "paste"
            st.success("Dictated text received.")
            st.rerun()
    else:
        st.info("Microphone unavailable in this version.")

    with st.form("article_form"):
        article = st.text_area(
            T["paste"],
            key="article",
            height=220,
            label_visibility="collapsed",
            placeholder=T["paste"]
        )
        analyze_submitted = st.form_submit_button(T["analyze"], use_container_width=True)

if article.strip() != previous_article.strip():
    st.session_state.article_source = "paste"

source_label = T["manual_paste"] if st.session_state.get("article_source") == "paste" else T["loaded_url_source"]
st.caption(f"{T['text_source']} : {source_label}")

if st.session_state.get("loaded_url"):
    st.caption(f"URL : {st.session_state.loaded_url}")


# -----------------------------
# Main analysis
# -----------------------------
if analyze_submitted:
    st.session_state.last_result = analyze_article(article)
    st.session_state.last_article = article

result = st.session_state.last_result
article_for_analysis = st.session_state.last_article

if result:
    col1, col2, col3 = st.columns(3)
    col1.metric(T["classic_score"], result["M"], help=T["help_classic_score"])
    col2.metric(T["improved_score"], result["improved"], help=T["help_improved_score"])
    col3.metric(T["hard_fact_score"], result["hard_fact_score"], help=T["help_hard_fact_score"])

    score = result["hard_fact_score"]
    if score <= 6:
        couleur, etiquette, message = "🔴", T["fragile"], T["fragile_message"]
    elif score <= 11:
        couleur, etiquette, message = "🟠", T["doubtful"], T["doubtful_message"]
    elif score <= 15:
        couleur, etiquette, message = "🟡", T["plausible"], T["plausible_message"]
    else:
        couleur, etiquette, message = "🟢", T["robust"], T["robust_message"]

    st.subheader(f"{couleur} {T['credibility_gauge']} : {etiquette}")
    st.progress(score / 20)
    st.caption(f"{T['score']} : {score}/20 — {message}")
    st.caption("On this scale, a truly credible text generally belongs in the robust zone.")

    st.subheader("Cognitive diagnosis")
    life_score = round((result["hard_fact_score"] / 20) * 100, 1)
    mecroyance_bar = max(0.0, min(1.0, (result["M"] + 10) / 30))

    col1, col2 = st.columns(2)
    with col1:
        st.write("Cognitive vitality")
        st.progress(life_score / 100)
        st.caption(f"{life_score}%")
    with col2:
        st.write("Mécroyance index")
        st.progress(mecroyance_bar)
        st.caption(f"M = {result['M']}")

    st.subheader(f"{T['verdict']} : {result['verdict']}")
    st.subheader(T["summary"])

    m1, m2 = st.columns(2)
    m1.metric("G — gnōsis", result["G"])
    m2.metric("N — nous", result["N"])
    m3, m4 = st.columns(2)
    m3.metric("D — doxa", result["D"])
    m4.metric("V — verifiability", result["V"])
    m5, m6 = st.columns(2)
    m5.metric("QS", result["source_quality"])
    m6.metric("RC", round(result["avg_claim_risk"], 1))
    m7, m8 = st.columns(2)
    m7.metric("VC", round(result["avg_claim_verifiability"], 1))
    m8.metric("F", len(result["red_flags"]))

    st.divider()
    st.subheader("G-N-D cognitive triangle")
    st.caption("The text is placed in the space of cognition: articulated knowledge, integrated understanding, and assertive certainty.")
    fig_triangle = plot_cognitive_triangle_3d(result["G"], result["N"], result["D"])
    st.pyplot(fig_triangle, use_container_width=True)

    st.subheader("Cognitive metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mécroyance index (M)", round(result["M"], 2))
    with col2:
        st.metric("Lie index (ME)", round(result["ME"], 2))

    delta_mm = round(result["M"] - result["ME"], 2)
    st.caption(f"Cognitive gap (M − ME): {delta_mm}")

    if result["M"] > result["ME"] + 1:
        dominant_pattern = "Dominant structure: mécroyance"
    elif result["ME"] > result["M"] + 1:
        dominant_pattern = "Dominant structure: strategic lying"
    else:
        dominant_pattern = "Dominant structure: mixed or ambiguous"

    st.subheader("Dominant cognitive structure")
    st.write(dominant_pattern)

    if result["ME"] > result["M"] and result["ME"] > 0:
        cognitive_type = "Possible strategic lying"
    elif result["M"] < 0:
        cognitive_type = "Strong mécroyance / cognitive closure"
    else:
        cognitive_type = "Probably sincere but misaligned cognition"

    st.subheader("Cognitive interpretation")
    st.write(cognitive_type)

    if result["M"] - result["ME"] > 3:
        diagnosis = "Strong mécroyance structure"
    elif result["M"] > result["ME"]:
        diagnosis = "Moderate mécroyance structure"
    elif abs(result["M"] - result["ME"]) <= 1:
        diagnosis = "Ambiguous cognitive structure"
    else:
        diagnosis = "Possible strategic deception"

    st.subheader("Cognitive diagnosis")
    st.write(diagnosis)

    gauge_value, gauge_label, gauge_color, ME_gauge = compute_lie_gauge(
        result["G"], result["N"], result["D"], result["M"]
    )

    st.write("Cognitive tension (mécroyance vs lying)")
    st.caption(
        "This gauge indicates whether the discourse looks more like a sincere error "
        "(mécroyance) or a possible manipulation. "
        "The further the gauge progresses, the closer the structure gets to lying."
    )

    st.markdown(f"""
    <div style="width:100%; margin-top:10px; margin-bottom:10px;">
        <div style="
            width:100%;
            height:26px;
            background:#e5e7eb;
            border-radius:12px;
            overflow:hidden;
            border:1px solid #cbd5e1;
        ">
            <div style="
                width:{gauge_value*100}%;
                height:100%;
                background:{gauge_color};
                transition:width 0.4s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<b style='color:{gauge_color}'>{gauge_label}</b> — {round(gauge_value*100,1)}%",
        unsafe_allow_html=True
    )

    st.caption("Sincere error ⟵⟶ Probable manipulation")

    with st.expander(T["strengths_detected"], expanded=True):
        if result["strengths"]:
            for item in result["strengths"]:
                st.success(item)
        else:
            st.info(T["few_strong_signals"])

    with st.expander(T["weaknesses_detected"], expanded=True):
        if result["weaknesses"]:
            for item in result["weaknesses"]:
                st.error(item)
        else:
            st.success(T["no_major_weakness"])

    st.divider()
    st.subheader("Cognitive structure of the analyzed text")
    st.info(T["llm_intro"])

    cog = Cognition(result["G"], result["N"], result["D"])
    overconfidence = result["D"] - (result["G"] + result["N"])
    calibration = result["D"] / (result["G"] + result["N"]) if (result["G"] + result["N"]) > 0 else 10
    revisability = (result["G"] + result["N"] + result["V"]) - result["D"]
    closure = (result["D"] * (1 + len(result["red_flags"]) / 5)) / (result["G"] + result["N"]) if (result["G"] + result["N"]) > 0 else 10

    c1, c2 = st.columns(2)
    c1.metric(T["overconfidence"], round(overconfidence, 2))
    c2.metric(T["calibration"], round(calibration, 2))
    c3, c4 = st.columns(2)
    c3.metric(T["revisability"], round(revisability, 2))
    c4.metric(T["cognitive_closure"], round(closure, 2))
    st.markdown(f"**{T['interpretation']} :** {cog.interpret()}")

    st.subheader(T["hard_fact_checking_by_claim"])
    claims_df = pd.DataFrame(
        [
            {
                T["claim"]: c.text,
                T["status"]: c.status,
                f"{T['verifiability']} /20": c.verifiability,
                f"{T['risk']} /20": c.risk,
                T["number"]: T["yes"] if c.has_number else T["no"],
                T["date"]: T["yes"] if c.has_date else T["no"],
                T["named_entity"]: T["yes"] if c.has_named_entity else T["no"],
                T["attributed_source"]: T["yes"] if c.has_source_cue else T["no"],
            }
            for c in result["claims"]
        ]
    )

    if not claims_df.empty:
        st.dataframe(claims_df, use_container_width=True, hide_index=True)
    else:
        st.info(T["paste_longer_text"])

    st.divider()
    st.subheader(T["ai_module"])
    st.caption(T["ai_module_caption"])

    if client is None:
        st.warning(T["ai_unavailable"])
    else:
        if st.button(T["generate_ai_analysis"], key="generate_ai_analysis"):
            with st.spinner("AI analysis in progress..."):
                ai_summary = generate_ai_summary(article_for_analysis, result)
            st.subheader(T["ai_analysis_result"])
            st.markdown(ai_summary)

    if st.session_state.get("article_source") == "paste":
        st.divider()
        st.subheader(T["external_corroboration_module"])
        st.caption(T["external_corroboration_caption"])
        with st.spinner(T["corroboration_in_progress"]):
            corroboration = corroborate_claims(article_for_analysis, max_claims=5, max_results_per_claim=3)
        if corroboration:
            for i, item in enumerate(corroboration, start=1):
                title_preview = item["claim"][:140] + ("..." if len(item["claim"]) > 140 else "")
                with st.expander(f"{T['claim']} {i} : {title_preview}", expanded=(i == 1)):
                    st.markdown(f"**{T['corroboration_verdict']} :** {display_corroboration_verdict(item['verdict'])}")
                    st.markdown(f"**{T['generated_query']} :** `{item['query']}`")
                    if item["matches"]:
                        for match in item["matches"]:
                            st.markdown(f"**[{match['title']}]({match['url']})**")
                            st.markdown(
                                f"- **{T['match_score']}** : {match['match_score']['score']}\n"
                                f"- **{T['contradiction_signal']}** : {T['detected'] if match['match_score']['contradiction_signal'] else T['not_detected']}"
                            )
                            if match["snippet"]:
                                st.caption(match["snippet"])
                    else:
                        st.warning(T["no_strong_sources_found"])
        else:
            st.info(T["no_corroboration_found"])
else:
    st.info(T["paste_text_or_load_url"])


# -----------------------------
# Method
# -----------------------------
if show_method:
    st.subheader(T["method"])
    st.markdown(
        f"### {T['original_formula']}\n"
        f"`M = (G + N) − D`\n"
        f"- {T['articulated_knowledge_density']}\n"
        f"- {T['integration']}\n"
        f"- {T['assertive_rigidity']}\n\n"
        f"### {T['llm_metrics']}\n"
        f"- **{T['overconfidence']}** : `D - (G + N)`\n"
        f"- **{T['calibration']}** : `D / (G + N)`\n"
        f"- **{T['revisability']}** : `(G + N + V) - D`\n"
        f"- **{T['cognitive_closure']}** : `(D * S) / (G + N)`\n\n"
        f"{T['disclaimer']}"
    )


# -----------------------------
# Interactive laboratory
# -----------------------------
st.divider()
st.subheader("Interactive mécroyance laboratory")
st.caption(
    "Experiment with the cognitive formula: M = (G + N) − D. "
    "Adjust the parameters to observe how cognitive stages evolve."
)

g_game = st.slider("G — gnōsis (articulated knowledge)", 0.0, 10.0, 5.0, 0.5)
n_game = st.slider("N — nous (lived integration)", 0.0, 10.0, 5.0, 0.5)
d_game = st.slider("D — doxa (certainty / saturation)", 0.0, 10.0, 5.0, 0.5)

m_game = round((g_game + n_game) - d_game, 1)

st.markdown(
    f"""
    <div style="
        background:#f1f5f9;
        border-radius:14px;
        padding:18px;
        margin-top:10px;
        border:1px solid #dbe3ec;
        text-align:center;
        font-size:1.3rem;
        font-weight:700;
    ">
        M = ({g_game:.1f} + {n_game:.1f}) − {d_game:.1f} =
        <span style="color:#0b6e4f;">{m_game:.1f}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if m_game < 0:
    stage = "Cognitive closure"
    explanation = "Certainty exceeds understanding: thought locks itself."
    percent = 10
elif m_game <= 4:
    stage = "Cognitive childhood"
    explanation = "An emerging cognitive structure, still fragile."
    percent = 25
elif m_game <= 10:
    stage = "Cognitive adolescence"
    explanation = "Stable cognition, but still unsettled."
    percent = 50
elif m_game <= 17:
    stage = "Cognitive maturity"
    explanation = "Balance between knowledge, experience, and doubt."
    percent = 75
elif m_game < 19:
    stage = "Structural wisdom"
    explanation = "A rare state of cognitive balance."
    percent = 90
else:
    stage = "Asymptote of truth"
    explanation = "Theoretical horizon of maximal coherence."
    percent = 100

st.markdown(f"**Current stage: {stage}**")
st.progress(percent / 100)
st.caption(f"M = {m_game} — {explanation}")

st.markdown("### Cognitive evolution")

stages = [
    ("Closure", -10, 0),
    ("Childhood", 0, 4.1),
    ("Adolescence", 4.1, 10.1),
    ("Maturity", 10.1, 17.1),
    ("Wisdom", 17.1, 19.1),
    ("Asymptote", 19.1, 21),
]

cols = st.columns(len(stages))
for i, (name, low, high) in enumerate(stages):
    active = low <= m_game < high
    with cols[i]:
        if active:
            st.success(name)
        else:
            st.info(name)

st.caption("When G and N increase without inflation of D, cognition gains revisability.")
