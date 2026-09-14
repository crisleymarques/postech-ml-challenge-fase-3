# Backlog e Débito Técnico

Este documento lista os pontos de melhoria e correções identificados no Code Review inicial (EDA e Pipeline) que devem ser priorizados nas próximas etapas do projeto. 

## 🟢 Melhorias (Próximas Etapas e Modelagem)

- [ ] **1. Substituir `print()` por `logging`**
  Necessário para facilitar a observabilidade na integração com Apache Airflow (Etapa 2).

- [ ] **2. Adicionar `pyproject.toml`**
  Tornar o repositório instalável (`pip install -e .`) para evitar manipulação de rotas com `sys.path.insert`.

- [ ] **3. Adicionar Cross-Validation ao benchmark**
  Garantir que as métricas dos classificadores não sejam enviesadas por um único *split* de dados.

- [ ] **4. Documentar exclusão do split original**
  Deixar claro na documentação por que o *split* original do dataset foi ignorado em favor do `train_test_split`.

- [ ] **5. Estratégias para a classe "Normal"**
  Considerar uso de técnicas como oversampling ou SMOTE para tratar a classe severamente desbalanceada (apenas 10.3%).

- [ ] **6. Avaliar `SGDClassifier`**
  Substituir `RandomForestClassifier` por `SGDClassifier` no benchmark, já que lida melhor com matrizes esparsas do TF-IDF.
