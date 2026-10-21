
"""Smart Keyboard AI — prédiction du prochain mot avec des modèles n-grammes."""

import streamlit as st
import pandas as pd

from src.pipeline import (
    NGramModel,
    InterpolatedModel,
    EOS,
    load_documents,
    split_by_document,
    backoff_top_k,
    top_k_accuracy,
)


# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Smart Keyboard AI",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DESIGN CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fc;
        color: #20243a;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171b35, #292d53);
    }

    [data-testid="stSidebar"] * {
        color: #f3f4ff;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #20243a;
    }

    h1, h2, h3 {
        color: #20243a;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e7eaf3;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(30, 40, 90, 0.04);
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid #dedff5;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: #5146c7;
        color: white;
        border: none;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: #5146c7;
        color: #5146c7;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        color: white;
        background: #4036aa;
    }

    [data-testid="stTabs"] button {
        font-weight: 600;
    }

    [data-testid="stTextArea"] textarea {
        border-radius: 12px;
        border: 1px solid #dfe3f0;
        background: white;
    }

    .hero {
        background: linear-gradient(120deg, #171b35, #5146c7);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(44, 44, 100, 0.13);
    }

    .hero-label {
        color: #d2d0ff;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 700;
    }

    .hero-title {
        color: white;
        font-size: 36px;
        font-weight: 800;
        margin: 10px 0;
    }

    .hero-description {
        color: #e5e7ff;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-card {
        background: white;
        border: 1px solid #e7eaf3;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .suggestion-note {
        color: #6c7189;
        font-size: 13px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. EN-TÊTE
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">NLP · LANGUAGE TECHNOLOGY</div>
        <div class="hero-title">⌨️ Smart Keyboard AI</div>
        <p class="hero-description">
            Intelligent next-word prediction powered by statistical
            language models and Hugging Face.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. CONFIGURATION LATÉRALE
# ============================================================

with st.sidebar:
    st.markdown("##  Configuration")
    st.caption("Personnalisez le modèle de prédiction.")

    smoothing = st.slider(
        "Lissage add-k",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Contrôle le lissage des probabilités.",
    )

    model_choice = st.selectbox(
        "Modèle de prédiction",
        [
            "Interpolation (3/2/1-gram)",
            "Back-off (3 → 2 → 1)",
            "Trigramme",
            "Bigramme",
            "Unigramme",
        ],
    )

    top_k = st.slider(
        "Nombre de suggestions",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.divider()
    st.markdown("### 📚 Corpus")
    st.caption("Source : Hugging Face")
    st.caption("Dataset : WikiText-2")
    st.caption("Langue : anglais")
    st.caption("Approche : modèles statistiques n-grammes")
    st.divider()
    st.caption("Smart Keyboard AI · Projet NLP")


# ============================================================
# 5. CHARGEMENT DU CORPUS
# ============================================================

@st.cache_resource(show_spinner="Chargement du corpus Hugging Face…")
def prepare():
    docs = load_documents(source="huggingface")
    train, test = split_by_document(docs, 0.2)

    models = {
        "unigram": NGramModel(train, n=1, k=0.1),
        "bigram": NGramModel(train, n=2, k=0.1),
        "trigram": NGramModel(train, n=3, k=0.1),
        "interpolated": InterpolatedModel(train, k=0.1),
    }

    return docs, train, test, models


try:
    docs, train, test, base_models = prepare()
except Exception as e:
    st.error(f"Impossible de charger le corpus Hugging Face : {e}")
    st.info(
        "Vérifiez votre connexion Internet et installez les dépendances "
        "avec : python -m pip install -r requirements.txt"
    )
    st.stop()


@st.cache_resource
def build_model(model_name, k_value, train_data):
    if model_name == "Unigramme":
        return NGramModel(train_data, n=1, k=k_value)

    if model_name == "Bigramme":
        return NGramModel(train_data, n=2, k=k_value)

    if model_name == "Trigramme":
        return NGramModel(train_data, n=3, k=k_value)

    if model_name == "Interpolation (3/2/1-gram)":
        return InterpolatedModel(train_data, k=k_value)

    return (
        NGramModel(train_data, n=3, k=k_value),
        NGramModel(train_data, n=2, k=k_value),
        NGramModel(train_data, n=1, k=k_value),
    )


selected = build_model(
    model_choice,
    smoothing,
    tuple(tuple(sentence) for sentence in train),
)


# ============================================================
# 6. STATISTIQUES DU CORPUS
# ============================================================

sentence_count = sum(len(sentences) for sentences in docs.values())
token_count = sum(
    len(sentence)
    for sentences in docs.values()
    for sentence in sentences
)

st.markdown("### 📊 Aperçu du corpus")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents", f"{len(docs):,}")

with col2:
    st.metric("Segments textuels", f"{sentence_count:,}")

with col3:
    st.metric("Tokens", f"{token_count:,}")


# ============================================================
# 7. ONGLETS
# ============================================================

tab_predict, tab_metrics, tab_about = st.tabs(
    [
        "✨ Prédiction",
        "📈 Évaluation",
        "ℹ️ À propos",
    ]
)


# ============================================================
# 8. FONCTION CALLBACK
# IMPORTANT : modifier le texte dans le callback, pas après
# la création du widget text_area.
# ============================================================

def add_selected_word():
    choice = st.session_state.get("smart_selected_word")

    if not choice or choice == EOS:
        return

    current_text = st.session_state.get("smart_text", "")

    st.session_state["smart_text"] = (
        current_text.rstrip() + " " + choice
    ).strip()


# ============================================================
# 9. ONGLET PRÉDICTION
# ============================================================

with tab_predict:
    st.markdown("## ✍️ Prédire le prochain mot")
    st.write(
        "Saisissez une phrase en anglais. Le modèle analysera "
        "son contexte et proposera les mots suivants les plus probables."
    )

    # Initialisation avant la création du widget.
    if "smart_text" not in st.session_state:
        st.session_state["smart_text"] = ""

    # Un seul champ de saisie : pas de sélecteur d'exemples.
    text_input = st.text_area(
        "Votre texte",
        key="smart_text",
        height=130,
        placeholder="Écrivez votre texte en anglais ici...",
        help="Saisissez quelques mots pour prédire le mot suivant.",
    )

    words = text_input.lower().split()
    words = [
        word.strip(".,!?;:\"()[]{}")
        for word in words
        if word.strip(".,!?;:\"()[]{}")
    ]

    st.caption(f"{len(text_input.split())} mot(s) saisi(s)")

    if words:
        if model_choice == "Back-off (3 → 2 → 1)":
            suggestions, used = backoff_top_k(
                *selected,
                words,
                top_k,
            )
            st.caption(f"Modèle effectivement utilisé : **{used}**")
        else:
            suggestions = selected.top_k(words, top_k)
            st.caption(f"Modèle sélectionné : **{model_choice}**")

        if suggestions:
            st.markdown("### 💡 Suggestions")

            for index, (word, probability) in enumerate(
                suggestions, start=1
            ):
                display_word = (
                    "Fin de phrase (</s>)"
                    if word == EOS
                    else word
                )

                st.progress(
                    min(float(probability), 1.0),
                    text=(
                        f"{index}. {display_word} — "
                        f"{probability:.2%}"
                    ),
                )

            st.markdown("### 📝 Construire la phrase")
            st.write("Sélectionnez le mot que vous souhaitez ajouter.")

            choice = st.radio(
                "Mot à ajouter",
                [word for word, _ in suggestions],
                format_func=lambda word: (
                    "Fin de phrase (</s>)"
                    if word == EOS
                    else word
                ),
                horizontal=True,
                key="smart_selected_word",
            )

            st.button(
                "➕ Ajouter le mot choisi à ma phrase",
                on_click=add_selected_word,
                type="primary",
                use_container_width=True,
            )

        else:
            st.warning(
                "Aucune suggestion disponible pour ce contexte."
            )

    else:
        st.info(
            "Saisissez quelques mots dans « Votre texte » "
            "pour afficher les suggestions."
        )

    st.caption(
        "Les probabilités sont estimées par un modèle statistique "
        "n-gramme. Elles ne proviennent pas d'un modèle neuronal."
    )


# ============================================================
# 10. ONGLET ÉVALUATION
# ============================================================

with tab_metrics:
    st.markdown("## 📈 Évaluation des modèles")

    st.write(
        "Les métriques sont calculées sur le jeu de test, composé "
        "des derniers 20 % des documents selon le découpage utilisé."
    )

    st.info(
        "La première évaluation peut prendre un peu de temps."
    )

    if st.button(
        "Calculer les métriques",
        type="primary",
        key="evaluate_models",
    ):
        rows = []

        with st.spinner("Calcul des métriques en cours…"):
            for name, model in base_models.items():
                perplexity, zero_share = model.perplexity(test)
                accuracy, _ = top_k_accuracy(model, test, 3)

                rows.append(
                    {
                        "Modèle": name,
                        "Perplexité (↓)": round(perplexity, 3),
                        "Tokens à probabilité nulle": (
                            f"{zero_share:.2%}"
                        ),
                        "Accuracy Top-3 (↑)": f"{accuracy:.2%}",
                    }
                )

        results = pd.DataFrame(rows)

        st.markdown("### Résultats")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Télécharger les résultats CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="smart_keyboard_evaluation.csv",
            mime="text/csv",
        )

        st.caption(
            "Une perplexité plus faible est généralement préférable. "
            "Une accuracy Top-3 plus élevée indique que le mot attendu "
            "apparaît plus souvent parmi les trois suggestions."
        )


# ============================================================
# 11. ONGLET À PROPOS
# ============================================================

with tab_about:
    st.markdown("## 📚 À propos de Smart Keyboard AI")

    st.markdown(
        """
        <div class="section-card">
            <h3>🧠 Objectif du projet</h3>
            <p>
                Smart Keyboard AI est une application de traitement
                automatique du langage naturel (NLP) qui prédit le
                prochain mot à partir du contexte d'une phrase.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔬 Modèles disponibles")

    st.markdown(
        """
        - **Unigramme :** estime les probabilités à partir des mots seuls.
        - **Bigramme :** exploite le mot précédent.
        - **Trigramme :** exploite les deux mots précédents.
        - **Interpolation :** combine plusieurs ordres de n-grammes.
        - **Back-off :** utilise un contexte plus court lorsque le contexte
          le plus long ne fournit pas de suggestion adaptée.
        """
    )

    st.markdown("### 📖 Corpus utilisé")

    st.markdown(
        """
        - **Source :** `Salesforce/wikitext` sur Hugging Face.
        - **Configuration :** `wikitext-2-raw-v1`.
        - **Langue :** anglais.
        - **Prétraitement :** réalisé par le pipeline du projet.
        - **Évaluation :** séparation des documents entre entraînement
          et test.
        """
    )

    st.warning(
        "Limite : les modèles n-grammes s'appuient sur les fréquences "
        "observées dans le corpus. Ils ne comprennent pas le sens "
        "d'une phrase comme un modèle Transformer."
    )


# ============================================================
# 12. PIED DE PAGE
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777d96; font-size:12px;">
        Smart Keyboard AI · NLP · Statistical Language Modeling
    </div>
    """,
    unsafe_allow_html=True,
)
