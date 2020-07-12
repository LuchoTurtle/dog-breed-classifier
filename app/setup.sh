mkdir -p ~/.streamlit

echo "[general
email = \n
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml




