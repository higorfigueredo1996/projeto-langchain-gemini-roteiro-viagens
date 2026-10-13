# Assistente de Roteiro de Viagens com LangChain — adaptado para Gemini

Projeto desenvolvido a partir do curso da Alura **"LangChain e Python: criando ferramentas com a LLM OpenAI"**, adaptado do zero para usar o **Google Gemini** em vez da API paga da OpenAI.

## Por que adaptar?

O curso original é construído inteiramente em cima da API da OpenAI, que é paga desde o primeiro uso. No fórum do curso, a comunidade sugeriu o Gemini como alternativa gratuita — decidi seguir essa sugestão e refazer todo o curso trocando o "motor" de LLM, aula por aula.

## O que o projeto cobre

O repositório contém 4 scripts, cada um demonstrando um conceito diferente do ecossistema LangChain:

| Arquivo | Conceito | O que faz |
|---|---|---|
| `main.py` | Prompt Templates, Output Parsers, cadeias LCEL | Sugere uma cidade, restaurantes e atividades culturais, encadeando 3 cadeias em sequência (a saída de uma alimenta a entrada da próxima) |
| `main_chat.py` | Memória de conversa | Simula um chat com um guia de viagens ("Sr. Passeios") que lembra do contexto entre perguntas, usando `RunnableWithMessageHistory` |
| `main_langgraph.py` | Orquestração com grafos | Roteia a pergunta do usuário entre dois assistentes especializados (praia ou montanha) usando `LangGraph`, com nós assíncronos e saída estruturada (`with_structured_output`) |
| `main_rag.py` | RAG (Retrieval-Augmented Generation) | Responde perguntas sobre 3 PDFs de termos de cartão de crédito (Standard/Gold/Platinum), recuperando apenas os trechos relevantes via busca por similaridade (FAISS) |

## Principais adaptações (OpenAI → Gemini)

Trocar o provedor de LLM em cada arquivo seguiu, na maior parte das vezes, o mesmo padrão:

```python
# Antes
from langchain_openai import ChatOpenAI
modelo = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=api_key)

# Depois
from langchain_google_genai import ChatGoogleGenerativeAI
modelo = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.5, api_key=api_key)
```

Essa é, na prática, a demonstração mais concreta do que o curso promete: o LangChain desacopla a aplicação do provedor de LLM específico — meia dúzia de linhas mudam, o resto da lógica (prompts, parsers, cadeias, memória, roteamento) permanece intacto.

## Desafios reais enfrentados (além do que o curso previa)

Adaptar para um provedor diferente do original trouxe problemas que não estavam no roteiro — e que acabaram sendo o aprendizado mais valioso do projeto:

- **Compatibilidade de ambiente**: o Python 3.14 (mais recente na época) quebrava a instalação de dependências do LangChain por falta de wheels pré-compilados (exigia Visual C++ Build Tools). Resolvido criando um ambiente virtual dedicado com Python 3.12.
- **Modelos descontinuados em tempo real**: `gemini-2.0-flash` e `text-embedding-004` deixaram de existir durante o desenvolvimento, exigindo migração para `gemini-3.6-flash` e `gemini-embedding-001`.
- **Limite de cota gratuita (TPM)**: gerar embeddings dos 3 PDFs de uma vez estourava o limite de 30 mil tokens/minuto da conta gratuita do Gemini. Solução: processamento em **lotes** com pausa entre eles, populando o índice FAISS incrementalmente.
- **Qualidade de recuperação (RAG)**: com poucos trechos recuperados por busca (`k` baixo), o sistema só encontrava informações do PDF "Gold", ignorando o "Platinum" — mesmo ele estando corretamente indexado. Aumentar o `k` do retriever resolveu, evidenciando como documentos tematicamente parecidos competem entre si na busca por similaridade.

## Stack

`Python 3.12` · `LangChain` · `LangGraph` · `Google Gemini API` (`langchain-google-genai`) · `FAISS` · `PyPDF`

## Como rodar

> **Atenção ao `main_rag.py`:** ele lê três PDFs da pasta `documentos/`
> (`GTB_standard_Nov23.pdf`, `GTB_gold_Nov23.pdf` e `GTB_platinum_Nov23.pdf`),
> que são material do curso da Alura e por isso não estão neste repositório.
> Crie a pasta `documentos/` e coloque ali seus próprios PDFs (ajustando os
> nomes na lista `arquivos` do script), ou use os arquivos do repositório do curso.

```bash
python -m venv venv
.\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz com sua chave do [Google AI Studio](https://aistudio.google.com/apikey):

```
GOOGLE_API_KEY="sua_chave_aqui"
```

Depois, rode qualquer um dos scripts:

```bash
python main.py
python main_chat.py
python main_langgraph.py
python main_rag.py
```

---

Projeto de estudo desenvolvido durante minha transição de carreira para IA e automação.
