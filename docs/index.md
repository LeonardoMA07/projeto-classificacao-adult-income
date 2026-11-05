# Classificação da Faixa de Renda de Adultos

**Etapa 1:** Análise Exploratória de Dados (EDA) e pré-processamento · **Etapa 2:** Classificação

| | |
|---|---|
| **Aluno** | Leonardo Moretti Alva |
| **Dataset** | [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census), Kaggle |
| **Etapa 1** | [EDA e pré-processamento](projeto_adult_income_eda.ipynb), com código, figuras e interpretações ([`.ipynb` no GitHub](https://github.com/LeonardoMA07/projeto-classificacao-adult-income/blob/main/docs/projeto_adult_income_eda.ipynb)) |
| **Etapa 2** | [Classificação](projeto_adult_income_classificacao.ipynb): modelos, avaliação, interpretação e discussão crítica ([`.ipynb` no GitHub](https://github.com/LeonardoMA07/projeto-classificacao-adult-income/blob/main/docs/projeto_adult_income_classificacao.ipynb)). Entrega: 04/11/2026 |

## Objetivo

Prever se a renda anual de uma pessoa é maior que 50 mil dólares (`>50K`) ou não (`<=50K`) a partir de dados demográficos e profissionais do Censo dos EUA de 1994. É um problema de classificação binária.

A Etapa 1 cobre a análise exploratória e a construção do pipeline de pré-processamento. A Etapa 2 usa esse mesmo pipeline para treinar, ajustar, avaliar e interpretar os modelos de classificação.

## Etapa 1: EDA e pré-processamento

### Os dados

- 31.947 registros e 11 features: 3 numéricas (`age`, `fnlwgt`, `education_num`) e 8 categóricas (`workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`).
- Alvo desbalanceado na proporção 3:1: 76% das pessoas ganham `<=50K`.
- Valores ausentes, marcados com `?`, em `workclass` e `occupation` (cerca de 5,6% cada) e `native_country` (25 linhas). Em `workclass` e `occupation` faltam nas mesmas linhas e marcam pessoas sem vínculo de trabalho.
- Inconsistência: 1.462 linhas trazem o texto "occupation" no lugar da ocupação. Esse grupo tem cerca de 53% de renda alta, o dobro da base, e por isso foi mantido como categoria própria.
- 65 linhas duplicadas, removidas.

### Principais achados

| Achado | Evidência |
|---|---|
| A escolaridade é a numérica mais ligada à renda | Correlação de 0,34 com o alvo; a taxa de `>50K` vai de 16% em *HS-grad* para 74% em *Doctorate* |
| Idade também importa | Correlação de 0,23; mediana de 44 anos em `>50K` contra 34 em `<=50K` |
| Estado civil separa bem as classes | 45% dos casados ganham `>50K`, contra 5% dos solteiros; parte do efeito vem da idade |
| `fnlwgt` é irrelevante | Correlação de -0,01 com a renda; é o peso amostral do Censo, não uma característica da pessoa |
| `education` e `education_num` são redundantes | Cada nível de escolaridade corresponde a exatamente um código |
| Viés de sexo | 31% dos homens ganham `>50K`, contra 11% das mulheres, mesmo com escolaridade mediana maior entre as mulheres |
| As numéricas separam só parcialmente as classes | No PCA, cada componente explica cerca de um terço da variância e as classes se sobrepõem |

### Estratégia de pré-processamento

| Etapa | Decisão | Motivo |
|---|---|---|
| Remoção de features | Descartar `education` e `fnlwgt` | Redundância e ausência de relação com o alvo |
| Valores ausentes | Categoria `Desconhecido` | A ausência carrega informação: só 10% desse grupo tem renda alta. A moda daria um emprego a quem não tem |
| Outliers | Manter todos | São idosos e níveis baixos de escolaridade, grupos reais |
| Encoding | One-Hot Encoding | Categorias sem ordem; *label encoding* criaria uma ordem falsa |
| Escala | `StandardScaler` | Coloca `age` e `education_num` na mesma escala para os modelos sensíveis a distância |

Tudo fica dentro de um `Pipeline` com `ColumnTransformer` do scikit-learn, ajustado só no conjunto de treino (80/20, estratificado). A saída tem 89 colunas.

### Recomendações para a etapa de modelagem

- Comparar Regressão Logística, Árvore de Decisão e Random Forest com validação cruzada estratificada.
- Avaliar com precisão, recall, F1-score e ROC-AUC, porque a acurácia engana com classes 3:1.
- Usar `class_weight='balanced'` e comparar as métricas entre homens e mulheres, por causa do viés encontrado.

## Etapa 2: Classificação

O [notebook da Etapa 2](projeto_adult_income_classificacao.ipynb) traz todo o código, as figuras e as justificativas. Resumo:

### Como os modelos foram comparados

- Mesmos dados, mesma limpeza, mesmo pipeline e mesma divisão treino/teste (80/20, estratificada) da Etapa 1.
- **Métrica:** F1-score da classe `>50K`. Com classes 3:1, a acurácia engana: o Dummy acerta 76% sem encontrar ninguém de renda alta.
- **Validação:** validação cruzada estratificada de 5 folds no treino. O conjunto de teste foi usado uma única vez, no final.
- `class_weight='balanced'` nos modelos e busca de hiperparâmetros com `GridSearchCV`.

### Resultados na validação cruzada

| Modelo | F1 (média de 5 folds) |
|---|---|
| Dummy (sempre `<=50K`) | 0,000 |
| Regressão Logística com regularização L2, ajustada (`C = 10`) | 0,654 |
| Random Forest padrão | 0,624 |
| Random Forest ajustada (`min_samples_leaf = 5`) | 0,659 |
| **Random Forest + categorias raras agrupadas** | **0,663** |

- **Hiperparâmetro decisivo:** `min_samples_leaf`. Com o padrão (1), as árvores decoram o treino e a Random Forest perde para a Regressão Logística.
- **Mudança no pipeline:** agrupar as categorias com menos de 100 pessoas no treino (`OneHotEncoder(min_frequency=100)`). São 42 categorias raras, 38 delas países, e a entrada cai de 89 para 51 colunas, com F1 igual ou um pouco melhor.
- **Curva de aprendizado:** treino e validação se estabilizam perto de 0,66–0,69. Mais linhas não ajudariam; o limite está nas features disponíveis.

### Melhor modelo no conjunto de teste

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 0,780 | 0,527 | 0,831 | 0,645 | 0,885 |

- O modelo encontra 83% das pessoas com renda `>50K`, ao custo de muitos falsos positivos (1.143, contra 259 falsos negativos), efeito do `class_weight='balanced'`.
- **Viés de sexo:** o recall é de 0,67 para mulheres e de 0,86 para homens.

### Interpretação

| Método | O que mostrou |
|---|---|
| Coeficientes da Regressão Logística | Os maiores são de categorias raríssimas (ex.: `native_country_Columbia`, com 45 pessoas), instáveis e não importantes. Entre as frequentes, ser casado multiplica a chance de `>50K` por cerca de 5 |
| Importância da Random Forest | `marital_status_Married-civ-spouse`, `education_num` e `age` |
| SHAP | Mesmas features no topo, com sentido: casado, mais escolaridade e mais idade aumentam a probabilidade; ser mulher reduz. Três predições individuais explicadas, incluindo o erro mais confiante do modelo |

### Discussão crítica

- **Modelo:** reproduz o viés de sexo, e retirar a coluna `sex` não resolveria porque `relationship` (`Husband`/`Wife`) revela o sexo; as probabilidades não são calibradas.
- **Dados:** esta versão do Kaggle não tem ganho de capital nem horas trabalhadas, presentes no *Adult* original; os dados são de 1994.
- **Operação:** publicar o pipeline completo versionado, com *model card*; monitorar dados e métricas por grupo e retreinar.

## Como reproduzir

```bash
git clone https://github.com/LeonardoMA07/projeto-classificacao-adult-income.git
cd projeto-classificacao-adult-income
pip install -r requirements.txt
jupyter notebook docs/projeto_adult_income_eda.ipynb            # Etapa 1
jupyter notebook docs/projeto_adult_income_classificacao.ipynb  # Etapa 2
```

O arquivo `adult income1.csv` já está na pasta `docs/`, ao lado dos notebooks. Basta executar todas as células. O notebook da Etapa 2 leva poucos minutos, por causa das buscas de hiperparâmetros e do SHAP.

## Estrutura do repositório

```
docs/
  index.md                                  # esta página
  projeto_adult_income_eda.ipynb            # Etapa 1: EDA e pré-processamento
  projeto_adult_income_classificacao.ipynb  # Etapa 2: classificação
  adult income1.csv                         # dataset do Kaggle
mkdocs.yml                        # configuração deste site
requirements.txt                  # bibliotecas usadas
```

## Referências

- Becker, B. e Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20) — versão usada aqui: [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census), Kaggle.
