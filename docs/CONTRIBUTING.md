# Guia de Contribuição, CI/CD e Definition of Done

Este documento descreve as regras para desenvolvimento, testes e aprovação de novos recursos no repositório.

## 1. Convenção de Commits

Utilizamos [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) para padronizar o histórico e facilitar a leitura das alterações. Todo commit deve seguir a seguinte estrutura:

```
<tipo>: <descrição curta no imperativo>
```

**Tipos permitidos:**
- `feat`: Uma nova funcionalidade ou recurso (ex: novos endpoints na API, novo modelo no Airflow).
- `fix`: Correção de bug.
- `docs`: Mudanças exclusivas em documentação (ex: atualizações no README).
- `test`: Adição ou correção de testes (sem alterar código de produção).
- `refactor`: Mudança de código que não corrige um bug nem adiciona uma feature (ex: renomear variáveis).
- `chore`: Atualização de tarefas de build, configuração de pacotes (ex: modificar `requirements.txt` ou `docker-compose.yml`).
- `perf`: Mudanças de código focadas em melhorar a performance (ex: otimização da latência).

**Exemplos válidos:**
- `feat: adicionar endpoint de health check`
- `fix: corrigir fallback para caso modelo não exista`
- `docs: atualizar README com instruções do Docker`

---

## 2. GitHub Actions (CI / CD)

O workflow de Integração Contínua encontra-se em `.github/workflows/ci.yml`. Ele roda automaticamente a cada push ou pull request aberto para a branch principal, garantindo a qualidade do código.

O pipeline possui **4 jobs paralelos/sequenciais**:

| Job | Descrição |
|:----|:----------|
| **lint-check** | Verificação de sintaxe Python (`py_compile`) de todos os módulos: source + pipelines + app + tests |
| **test** | Testes unitários com pytest em Python 3.10, 3.11, 3.12 (Ubuntu + 3.11 Windows) |
| **pipeline-check** | Smoke test de carregamento dataset, criação TF-IDF, criação de classificador, clean_text |
| **api-check** | 🔍 Validação da API: criação do app, rotas, 28 testes de API e teste E2E `/predict` com modelo dummy |

Para que o código seja integrado na master, **todos os jobs devem passar com sucesso**.

---

## 3. Definition of Done (DoD)

Para que uma Pull Request seja considerada concluída e pronta para merge, ela precisa atender aos seguintes critérios:

- [ ] **Código executando sem erros**: Todas as novas funcionalidades devem rodar localmente sem stack traces ou crashes inesperados.
- [ ] **Testes unitários/integrados adicionados e passando**: Qualquer nova rota de API ou método do pipeline deve estar coberto por testes (`pytest`).
- [ ] **Aprovação na CI/CD**: O workflow do GitHub Actions deve estar verde (sucesso em todos os jobs e versões de Python).
- [ ] **Documentação atualizada**: Quaisquer alterações na execução devem ser refletidas nos arquivos `docs/` e no `README.md`.
- [ ] **Código limpo e seguindo padrões**: Seguir as convenções de lint, PEP-8 e não deixar variáveis ou prints "perdidos" de debug no código de produção.
- [ ] **Validação em ambiente limpo**: Conseguir rodar os comandos do README do zero e tudo funcionar (Docker, Airflow, API, testes).
