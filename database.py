import psycopg # postgres driver for python
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DB_PORT = os.getenv("DB_PORT")

DB_URI = os.getenv("DATABASE_URL", f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{DB_PORT}/{POSTGRES_DB}")

def init_db():
    """Initialize custom tables for Streamlit UI thread history."""
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id VARCHAR(255) PRIMARY KEY,
                    title VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    thread_id VARCHAR(255) REFERENCES chat_threads(thread_id),
                    role VARCHAR(50),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()

def create_thread(thread_id: str, title: str):
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_threads (thread_id, title) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (thread_id, title)
            )
        conn.commit()

def get_all_threads():
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT thread_id, title FROM chat_threads ORDER BY created_at DESC")
            return cur.fetchall()

def add_message(thread_id: str, role: str, content: str):
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, role, content)
            )
        conn.commit()

def get_messages(thread_id: str):
    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM chat_messages WHERE thread_id = %s ORDER BY created_at ASC",
                (thread_id,)
            )
            return cur.fetchall()