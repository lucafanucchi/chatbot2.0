import os
import datetime
import json
import locale
import psycopg2
from contextlib import contextmanager

from flask import Flask, request, Response

# Imports do LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

# Imports do Google
from google.oauth2.service_account import Credentials # <-- ESTA LINHA ESTAVA FALTANDO
from googleapiclient.discovery import build

# --- Configuração de Localidade para Datas em Português ---
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    print("[!] Aviso: Localidade 'pt_BR.UTF-8' não encontrada. Usando a localidade padrão.")

# --- Inicialização do Flask App ---
app = Flask(__name__)

# --- CONFIGURAÇÕES E INICIALIZAÇÃO DAS APIS ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")
DATABASE_URL = os.getenv("DATABASE_URL")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/calendar"]

try:
    CREDS = Credentials.from_service_account_file("/etc/secrets/google_credentials.json", scopes=SCOPES)
    sheet = gspread.authorize(CREDS).open_by_key(SHEET_ID).sheet1
    calendar_svc = build("calendar", "v3", credentials=CREDS)
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_KEY, temperature=0.3)
except Exception as e:
    print(f"[!] Erro crítico na inicialização: {e}")
    llm = None

# --- MEMÓRIA DAS CONVERSAS (Agora gerenciada pelo banco de dados) ---
# O dicionário em memória servirá como um cache rápido, mas a fonte da verdade é o DB.
chat_histories = {}

# --- FUNÇÕES DE BANCO DE DADOS ---

@contextmanager
def get_db_connection():
    """Cria e gerencia uma conexão com o banco de dados PostgreSQL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def load_user_data(sender_id):
    """Carrega os dados e o histórico de um usuário do banco de dados."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT nome, cpf, chat_history FROM usuarios WHERE sender_id = %s", (sender_id,))
            user = cur.fetchone()
            if user:
                nome, cpf, history_json = user
                history_list = []
                if history_json:
                    for msg in history_json:
                        if msg.get('type') == 'ai':
                            history_list.append(AIMessage(content=msg.get('content')))
                        else:
                            history_list.append(HumanMessage(content=msg.get('content')))
                return {"nome": nome, "cpf": cpf}, history_list
    return None, []

def save_user_data(sender_id, user_data, chat_history):
    """Salva ou atualiza os dados e o histórico de um usuário no banco de dados."""
    history_json = json.dumps([{"type": msg.type, "content": msg.content} for msg in chat_history])
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO usuarios (sender_id, nome, cpf, chat_history, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (sender_id) DO UPDATE SET
                    nome = EXCLUDED.nome,
                    cpf = EXCLUDED.cpf,
                    chat_history = EXCLUDED.chat_history,
                    updated_at = NOW();
            """, (sender_id, user_data.get('nome'), user_data.get('cpf'), history_json))
            conn.commit()

# --- FERRAMENTAS DO AGENTE (AGENT TOOLS) ---

@tool
def verificar_disponibilidade_agenda(data_hora_iso: str) -> str:
    """Verifica se um horário específico está disponível na agenda. Use esta ferramenta ANTES de confirmar qualquer agendamento. Recebe uma data e hora no formato ISO 8601 (ex: '2025-06-25T14:00:00'). Retorna 'Horário disponível' ou 'Horário ocupado'."""
    try:
        tz_sp = datetime.timezone(datetime.timedelta(hours=-3))
        start_time = datetime.datetime.fromisoformat(data_hora_iso).astimezone(tz_sp)
        end_time = start_time + datetime.timedelta(minutes=30)
        eventos = calendar_svc.events().list(
            calendarId=GOOGLE_CALENDAR_ID, timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(), singleEvents=True
        ).execute().get('items', [])
        return "Horário ocupado" if eventos else "Horário disponível"
    except Exception as e:
        return f"Erro ao verificar a agenda: {str(e)}"

@tool
def registrar_consulta(nome_completo: str, cpf: str, data_hora_iso: str) -> str:
    """Registra e agenda a consulta no Google Calendar e Google Sheets. Use esta ferramenta APENAS como passo final, após ter coletado e confirmado o nome completo, CPF e a data/hora com o usuário. Recebe o nome_completo (string), cpf (string com 11 dígitos) e data_hora_iso (string no formato ISO 8601). Retorna uma mensagem de sucesso ou uma mensagem de erro."""
    try:
        # Extrai o telefone do sender_id (assumindo que está no formato correto)
        telefone = users.get(cpf, {}).get("telefone", "Não informado")
        tz_sp = datetime.timezone(datetime.timedelta(hours=-3))
        start_time = datetime.datetime.fromisoformat(data_hora_iso).astimezone(tz_sp)
        end_time = start_time + datetime.timedelta(minutes=30)
        sheet.append_row([nome_completo, cpf, telefone, start_time.strftime("%d/%m/%Y %H:%M"), "Confirmado"])
        calendar_svc.events().insert(
            calendarId=GOOGLE_CALENDAR_ID,
            body={
                "summary": f"Consulta: {nome_completo}", "description": f"CPF: {cpf}\nTelefone: {telefone}",
                "start": {"dateTime": start_time.isoformat(), "timeZone": "America/Sao_Paulo"},
                "end": {"dateTime": end_time.isoformat(), "timeZone": "America/Sao_Paulo"},
            }
        ).execute()
        return f"Agendamento confirmado com sucesso para {start_time.strftime('%A, %d de %B às %H:%M')}."
    except Exception as e:
        return f"Erro ao registrar a consulta: {str(e)}"

# --- CONFIGURAÇÃO DO AGENTE LANGCHAIN ---
tools = [verificar_disponibilidade_agenda, registrar_consulta]

prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente virtual para a clínica da Dra. Maria. Seu nome é Gemini.
    Seu objetivo é agendar consultas de forma eficiente e amigável. Seja sempre educado e responda em português do Brasil.
    A data e hora atual é: {current_time}.

    Contexto do Usuário: {user_context}

    Regras de Conversa:
    1. Se o contexto indicar que você já conhece o usuário (nome e/ou CPF), cumprimente-o pelo nome. Não peça informações que você já possui, a menos que o usuário queira alterá-las.
    2. Se o usuário for novo, siga os passos para agendar uma consulta: obter nome completo, obter CPF (11 dígitos), e a data/hora desejada.
    3. Antes de agendar, sempre use a ferramenta `verificar_disponibilidade_agenda`.
    4. Após a verificação, confirme TODOS os dados (nome, CPF, data/hora) com o usuário.
    5. Se o usuário confirmar, e somente neste caso, use a ferramenta `registrar_consulta`.
    6. Se não souber algo, peça para o usuário falar com um humano. Não invente informações.

    Formato da Resposta:
    Sempre formate sua resposta final como um objeto JSON contendo uma única chave "respostas", que é uma lista de strings. Cada string será enviada como uma mensagem separada para tornar a conversa mais natural.
    Exemplo: {{"respostas": ["Olá, Luca!", "Que bom te ver de novo. Como posso ajudar?"]}}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt) | JsonOutputParser()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- FUNÇÃO RESPONDER E ROTA WEBHOOK ---

def responder(mensagens, to):
    """Envia uma ou mais mensagens em TwiML."""
    response_xml = "<Response>"
    if isinstance(mensagens, str):
        mensagens = [mensagens]
    
    for msg in mensagens:
        if msg and msg.strip():
            print(f"[→ {to}] {msg}")
            msg_escaped = msg.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            response_xml += f"<Message><Body>{msg_escaped}</Body></Message>"
    
    response_xml += "</Response>"
    return Response(response_xml, mimetype="application/xml")

@app.route("/webhook", methods=["POST"])
def webhook():
    if not llm:
        return Response("Erro na inicialização do serviço de IA.", status=500)

    sender = request.form.get("From", "").replace("whatsapp:", "")
    text = request.form.get("Body", "").strip()
    print(f"[← {sender}] {text}")

    # Carrega dados e histórico do banco de dados
    user_data, chat_history = load_user_data(sender)
    user_context = json.dumps(user_data) if user_data else "Este é um novo usuário."

    # Invoca o agente
    try:
        now_br = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3)))
        current_time_str = now_br.strftime("%A, %d de %B de %Y, %H:%M")

        result = agent_executor.invoke({
            "input": text,
            "chat_history": chat_history,
            "current_time": current_time_str,
            "user_context": user_context
        })
        lista_de_respostas = result.get("respostas", ["Desculpe, não consegui processar minha resposta."])
    except Exception as e:
        print(f"[!] Erro ao invocar o agente: {e}")
        lista_de_respostas = ["Desculpe, ocorreu um erro interno. Tente novamente."]

    # Atualiza o histórico e salva no banco de dados
    resposta_concatenada = " ".join(lista_de_respostas)
    chat_history.extend([HumanMessage(content=text), AIMessage(content=resposta_concatenada)])
    
    # Precisamos garantir que os dados do usuário (nome/cpf) sejam salvos.
    # Uma abordagem mais avançada faria o próprio agente retornar os dados a serem salvos.
    # Por simplicidade, vamos salvar o que temos.
    if user_data is None: user_data = {}
    # Esta parte precisaria de uma lógica para extrair o nome/cpf da conversa e atualizar o user_data
    # antes de salvar. Por enquanto, o foco é a memória da conversa.
    save_user_data(sender, user_data, chat_history[-10:])

    return responder(lista_de_respostas, sender)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)