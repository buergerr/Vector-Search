#!/bin/bash

python -c "import nltk; nltk.download('stopwords')"

python preprocess_and_embed.py

python search_web_app_realtime.py
