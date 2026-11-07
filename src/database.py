
import sqlite3
import os
from datetime import datetime

DB_PATH = "outputs/news_history.db"


def create_table():
    os.makedirs("outputs", exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(news_text, prediction, confidence):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (news_text, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        news_text,
        prediction,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


def get_history():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, news_text, prediction, confidence, created_at
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    connection.close()

    return records
