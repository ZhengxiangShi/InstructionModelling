mkdir data/alpagasus && cd data/alpagasus
mkdir dolly && cd dolly
wget https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/filtered/dolly_3k.json
wget https://github.com/gpt4life/alpagasus/raw/main/data/filtered/chatgpt_9k.json
cd ..
mkdir alpaca && cd alpaca
wget https://raw.githubusercontent.com/gpt4life/alpagasus/main/data/filtered/claude_t45.json
cd ../../..
python src/reformat_alpagasus_data.py \
    --raw_data_dir data/alpagasus \
    --output_dir data/processed/alpagasus