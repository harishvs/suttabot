# Very rough beginnings of a bot which can answer questions based on Suttas:
TODO: work on a chatbot

## To set up a development env follow these steps


### download the 7B q8 model from Hugging face into models directory in root
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
```bash
mkdir models
cd models
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
cd ..
```


### download this data file into a data directory
```bash
mkdir data
cd data
wget https://www.ancient-buddhist-texts.net/Texts-and-Translations/Satipatthana/Satipatthana.pdf
cd ..
```

### create a venv
`python -m venv venv`

### activate the new env
`./venv/bin/activate`

### install required packages
`pip install -r requirements.txt`

### upgrade pip
`python -m pip install --upgrade pip`

## To load data 
`python load_files.py`

## To get answers based on the pdf in your data dir
`python question_answer_on_source.py`

Ref: 
https://swharden.com/blog/2023-07-30-ai-document-qa/
https://python.langchain.com/docs/use_cases/question_answering/



