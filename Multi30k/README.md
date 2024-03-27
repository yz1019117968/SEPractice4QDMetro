## Language Translation

### Dependencies  
```cmd
pip install torchtext==${与torchvision的版本相同}
pip install -U torchdata
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
pip install portalocker>=2.0.0
```

train and inference
```cmd
python train.py
```