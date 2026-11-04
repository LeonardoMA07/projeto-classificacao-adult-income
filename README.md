# Classificação da Faixa de Renda de Adultos

Projeto de Machine Learning com o dataset [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census), em duas etapas: análise exploratória (EDA) e pré-processamento (Etapa 1) e classificação (Etapa 2).

**Site do projeto:** https://leonardoma07.github.io/projeto-classificacao-adult-income/

**Notebooks:**

- Etapa 1: [docs/projeto_adult_income_eda.ipynb](docs/projeto_adult_income_eda.ipynb)
- Etapa 2: [docs/projeto_adult_income_classificacao.ipynb](docs/projeto_adult_income_classificacao.ipynb)

## Rodar o notebook

```bash
pip install -r requirements.txt
jupyter notebook docs/projeto_adult_income_eda.ipynb            # Etapa 1
jupyter notebook docs/projeto_adult_income_classificacao.ipynb  # Etapa 2
```

## Atualizar o site

```bash
mkdocs serve       # pré-visualiza em http://127.0.0.1:8000
mkdocs gh-deploy   # publica no GitHub Pages
```
