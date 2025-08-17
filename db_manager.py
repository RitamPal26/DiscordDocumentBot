# db_manager.py

import sqlite3
import logging
from typing import List, Tuple

DATABASE_FILE = 'plans.db'

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plans (
                message_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL,
                feature_request TEXT NOT NULL,
                original_text TEXT NOT NULL 
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                message_id INTEGER NOT NULL,
                original_id TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                assignee_id INTEGER,
                FOREIGN KEY (message_id) REFERENCES plans (message_id)
            )
        ''')
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")

def add_plan(message_id, channel_id, feature_request, original_text):
    """Adds a new plan to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO plans (message_id, channel_id, feature_request, original_text) VALUES (?, ?, ?, ?)",
                   (message_id, channel_id, feature_request, original_text))
    conn.commit()
    conn.close()

def add_task(task_id, message_id, original_id, description):
    """Adds a new task associated with a plan."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tasks (task_id, message_id, original_id, description) VALUES (?, ?, ?, ?)",
                   (task_id, message_id, original_id, description.strip()))
    conn.commit()
    conn.close()

def get_task_description(task_id: str) -> str | None:
    """Retrieves the description of a specific task."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT description FROM tasks WHERE task_id = ?", (task_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def update_task_status(task_id: str, status: str, assignee_id: int):
    """Updates the status and assignee of a task."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET status = ?, assignee_id = ? WHERE task_id = ?",
                   (status, assignee_id, task_id))
    conn.commit()
    conn.close()

def get_plan_and_tasks(message_id: int) -> Tuple[str, List[Tuple]]:
    """Retrieves the original plan text and all its associated tasks."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT original_text FROM plans WHERE message_id = ?", (message_id,))
    plan_text_result = cursor.fetchone()
    plan_text = plan_text_result[0] if plan_text_result else ""
    
    cursor.execute("SELECT original_id, description, status, assignee_id FROM tasks WHERE message_id = ?", (message_id,))
    tasks = cursor.fetchall()
    conn.close()
    return plan_text, tasks