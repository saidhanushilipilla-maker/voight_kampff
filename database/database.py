import sqlite3
from pathlib import Path
from datetime import datetime


# DATABASE PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_FOLDER = PROJECT_ROOT / "database"

DATABASE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_PATH = DATABASE_FOLDER / "voight_kampff.db"


# CREATE DATABASE CONNECTION

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection


# CREATE TABLE

def create_table():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            input_text TEXT NOT NULL,

            prediction TEXT NOT NULL,

            ai_probability REAL,

            human_probability REAL,

            perplexity REAL,

            burstiness REAL,

            timestamp TEXT NOT NULL,

            genai_explanation TEXT

        )
    """)

    connection.commit()

    connection.close()


# ---------------------------------------------------------
# SAVE DETECTION RESULT
# ---------------------------------------------------------

def save_detection(
    input_text,
    prediction,
    ai_probability,
    human_probability,
    perplexity,
    burstiness,
    genai_explanation
):

    connection = get_connection()

    cursor = connection.cursor()


    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    cursor.execute("""
        INSERT INTO detection_history (
            input_text,
            prediction,
            ai_probability,
            human_probability,
            perplexity,
            burstiness,
            timestamp,
            genai_explanation
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        input_text,

        prediction,

        ai_probability,

        human_probability,

        perplexity,

        burstiness,

        timestamp,

        genai_explanation

    ))


    connection.commit()

    connection.close()


# ---------------------------------------------------------
# GET ALL DETECTION HISTORY
# ---------------------------------------------------------

def get_detection_history():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute("""
        SELECT
            id,
            input_text,
            prediction,
            ai_probability,
            human_probability,
            perplexity,
            burstiness,
            timestamp,
            genai_explanation

        FROM detection_history

        ORDER BY id DESC
    """)


    rows = cursor.fetchall()

    connection.close()

    return rows


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("VOIGHT-KAMPFF DATABASE")
    print("=" * 60)


    # Create table

    create_table()

    print("\nDatabase created successfully.")

    print("Database location:")
    print(DATABASE_PATH)


    # Test record

    save_detection(

        input_text="Artificial intelligence is changing the world.",

        prediction="AI Generated",

        ai_probability=95.50,

        human_probability=4.50,

        perplexity=27.45,

        burstiness=0.42,

        genai_explanation="The text appears highly likely to be AI-generated based on the transformer detector and supporting linguistic signals."

    )


    print("\nTest detection saved successfully.")


    # Display history

    history = get_detection_history()


    print("\nDetection History")
    print("-" * 60)


    for row in history:

        print("\nID:", row[0])

        print("Text:", row[1])

        print("Prediction:", row[2])

        print("AI Probability:", row[3])

        print("Human Probability:", row[4])

        print("Perplexity:", row[5])

        print("Burstiness:", row[6])

        print("Timestamp:", row[7])

        print("Explanation:", row[8])


    print("\n" + "=" * 60)
    print("DATABASE TEST COMPLETED")
    print("=" * 60)