# Smart Keyboard AI — version Hugging Face

## Changements apportés
- Le corpus par défaut vient de Hugging Face : `Salesforce/wikitext`, configuration `wikitext-2-raw-v1` (anglais, texte Wikipédia).
- Prétraitement du texte et regroupement des lignes en documents.
- Interface Streamlit plus complète : choix du modèle (unigramme, bigramme, trigramme, interpolation ou back-off), réglage du lissage, nombre de suggestions configurable, probabilités visuelles, exemples et onglet d'évaluation.
- Le corpus synthétique d'origine reste dans `data/corpus.csv` comme option locale, mais il n'est plus le corpus utilisé par défaut.

## Installation (Windows)
Ouvrir CMD dans ce dossier :

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Le premier démarrage doit accéder à Internet pour télécharger WikiText-2. Le jeu est ensuite mis en cache par Hugging Face.

## Évaluation en ligne de commande
```bat
python -m src.evaluate
```
Cette commande utilise également le corpus Hugging Face et peut prendre un peu de temps.

## Organisation
- `app.py` : interface Streamlit
- `src/pipeline.py` : chargement du corpus, tokenisation, modèles n-grammes, lissage, interpolation, back-off et métriques
- `src/evaluate.py` : comparaison expérimentale des modèles
- `data/corpus.csv` : ancien corpus synthétique (optionnel)

## À savoir
WikiText-2 est un corpus anglais. Les suggestions ne sont pas celles d'un grand modèle neuronal : le projet garde les algorithmes n-grammes du TP, mais les entraîne sur un texte plus réaliste. Le découpage train/test se fait par documents pour réduire la fuite de données. Pour une utilisation hors ligne, il faut avoir téléchargé le dataset au moins une fois.
