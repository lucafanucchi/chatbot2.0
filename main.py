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
import gspread # <-- IMPORT CORRIGIDO
from google.oauth2.service_account import Credentials
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

llm, sheet, calendar_svc = None, None, None
try:
    CREDS = Credentials.from_service_account_file("/etc/secrets/google_credentials.json", scopes=SCOPES)
    sheet = gspread.authorize(CREDS).open_by_key(SHEET_ID).sheet1
    calendar_svc = build("calendar", "v3", credentials=CREDS)
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_KEY, temperature=0.3)
    print("[✔] Todos os serviços foram inicializados com sucesso.")
except Exception as e:
    print(f"[!] Erro crítico na inicialização: {e}")

# --- FUNÇÕES DE BANCO DE DADOS ---

@contextmanager
def get_db_connection():
    """Cria e gerencia uma conexão com o banco de dados PostgreSQL."""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    finally:
        if conn:
            conn.close()

def load_user_data(sender_id):
    """Carrega os dados e o histórico de um usuário do banco de dados."""
    user_data, chat_history = None, []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT nome, cpf, chat_history FROM usuarios WHERE sender_id = %s", (sender_id,))
                user = cur.fetchone()
                if user:
                    nome, cpf, history_json = user
                    user_data = {"nome": nome, "cpf": cpf}
                    if history_json:
                        for msg in history_json:
                            if msg.get('type') == 'ai':
                                chat_history.append(AIMessage(content=msg.get('content')))
                            else:
                                chat_history.append(HumanMessage(content=msg.get('content')))
    except Exception as e:
        print(f"[!] Erro ao carregar dados do usuário: {e}")
    return user_data, chat_history

def save_user_data(sender_id, user_data, chat_history):
    """Salva ou atualiza os dados e o histórico de um usuário no banco de dados."""
    try:
        history_json = json.dumps([{"type": msg.type, "content": msg.content} for msg in chat_history])
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO usuarios (sender_id, nome, cpf, chat_history, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (sender_id) DO UPDATE SET
                        nome = COALESCE(EXCLUDED.nome, usuarios.nome),
                        cpf = COALESCE(EXCLUDED.cpf, usuarios.cpf),
                        chat_history = EXCLUDED.chat_history,
                        updated_at = NOW();
                """, (sender_id, user_data.get('nome'), user_data.get('cpf'), history_json))
                conn.commit()
    except Exception as e:
        print(f"[!] Erro ao salvar dados do usuário: {e}")


# --- FERRAMENTAS DO AGENTE (AGENT TOOLS) ---

@tool
def verificar_disponibilidade_agenda(data_hora_iso: str) -> str:
    """Verifica se um horário específico está disponível na agenda. Use esta ferramenta ANTES de confirmar qualquer agendamento. Recebe uma data e hora no formato ISO 8601 (ex: '2025-06-25T14:00:00'). Retorna 'Horário disponível' ou 'Horário ocupado'."""
    try:
        tz_sp = datetime.timezone(datetime.timedelta(hours=-3))
        start_time = datetime.datetime.fromisoformat(data_hora_iso).astimezone(tz_sp)
        end_time = start_time + datetime.timedelta(minutes=30)
        
        if not (8 <= start_time.hour < 17 and start_time.weekday() < 5):
             return "Horário fora do expediente. O atendimento é de Segunda a Sexta, das 8h às 17h."

        eventos = calendar_svc.events().list(
            calendarId=GOOGLE_CALENDAR_ID, timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(), singleEvents=True
        ).execute().get('items', [])
        return "Horário ocupado" if eventos else "Horário disponível"
    except Exception as e:
        return f"Erro ao verificar a agenda: {str(e)}"

@tool
def registrar_consulta(nome_completo: str, cpf: str, data_hora_iso: str, telefone: str) -> str:
    """Registra e agenda a consulta no Google Calendar e Google Sheets. Use esta ferramenta APENAS como passo final, após ter coletado e confirmado o nome completo, CPF e a data/hora com o usuário. Recebe o nome_completo (string), cpf (string com 11 dígitos), data_hora_iso (string no formato ISO 8601) e telefone (string). Retorna uma mensagem de sucesso ou uma mensagem de erro."""
    try:
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
        return f"Agendamento confirmado com sucesso para {start_time.strftime('%A, %d de %B de %Y, às %H:%M')}."
    except Exception as e:
        return f"Erro ao registrar a consulta: {str(e)}"


# --- CONFIGURAÇÃO DO AGENTE LANGCHAIN ---
tools = [verificar_disponibilidade_agenda, registrar_consulta]

prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é Gemini, um assistente virtual para a clínica da Dra. Maria.
    Seu objetivo é agendar consultas de forma eficiente e amigável. Responda sempre em português do Brasil.
    A data e hora atual é: {current_time}.

    Informações sobre a clínica:
    - Horário de Atendimento: Segunda a Sexta, das 8h às 17h.
    - Você NÃO deve agendar fora desses horários. Informe ao usuário sobre o horário de funcionamento se ele tentar.

    Contexto do Usuário (se houver): {user_context}

    Regras de Conversa:
    1. Se o contexto indicar que você já conhece o usuário (nome e/ou CPF), cumprimente-o pelo nome. Não peça informações que você já possui, a menos que o usuário queira alterá-las.
    2. Se o usuário for novo, apresente-se e siga os passos para agendar: obter nome completo, obter CPF (11 dígitos), e a data/hora desejada.
    3. Antes de agendar, sempre use a ferramenta `verificar_disponibilidade_agenda` para garantir que o horário está livre e dentro do expediente.
    4. Após a verificação, confirme TODOS os dados (nome, CPF, data/hora) com o usuário.
    5. Se o usuário confirmar, e somente neste caso, use a ferramenta `registrar_consulta` para finalizar.
    6. Se não souber algo, peça para o usuário falar com um humano. Não invente informações.
    7. INSTRUÇÃO DE FORMATAÇÃO DE DATA: Quando você precisar apresentar uma data para o usuário, sempre o faça de forma amigável e totalmente em português do Brasil. Se uma ferramenta interna te retornar uma data em inglês (ex: 'Friday, June 20'), sua responsabilidade é traduzir e formatar essa data na sua resposta final para o usuário (ex: 'sexta-feira, 20 de junho').

    Formato da Resposta:
    Sempre formate sua resposta final como um objeto JSON contendo uma única chave "respostas", que é uma lista de strings. Cada string será enviada como uma mensagem separada para tornar a conversa mais natural.
    Exemplo: {{"respostas": ["Olá, Luca!", "Que bom te ver de novo. Como posso ajudar?"]}}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
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
        return responder("Desculpe, nosso sistema de inteligência artificial está temporariamente indisponível.", request.form.get("From", "").replace("whatsapp:", ""))

    sender = request.form.get("From", "").replace("whatsapp:", "")
    text = request.form.get("Body", "").strip()
    print(f"[← {sender}] {text}")

    user_data, chat_history = load_user_data(sender)
    if user_data is None:
        user_data = {}
        user_context = "Este é um novo usuário. Apresente-se e peça o nome completo."
    else:
        user_context = json.dumps(user_data)
        
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
        lista_de_respostas = ["Desculpe, ocorreu um erro interno. Por favor, tente novamente."]

    resposta_concatenada = " ".join(lista_de_respostas)
    chat_history.extend([HumanMessage(content=text), AIMessage(content=resposta_concatenada)])
    
    # Esta parte é uma simplificação. Uma versão mais avançada faria o agente
    # retornar os dados que ele extraiu para salvarmos de forma mais precisa.
    save_user_data(sender, user_data, chat_history[-10:])

    return responder(lista_de_respostas, sender)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)