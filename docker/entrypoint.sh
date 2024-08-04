#!/bin/bash
cd /app

if [ -f "config_template.py" ]; then
    cp config_template.py config.py
fi

python -c "import nltk; nltk.download('stopwords')"
python preprocess_and_embed.py
python search_web_app_realtime.py
