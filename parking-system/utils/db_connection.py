import mysql.connector
from mysql.connector import errorcode
import time

def get_db_connection(max_retries=3, retry_delay=2):
    """Get database connection with retry logic"""
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="root",
                database="spotcheck",
                autocommit=False,
                connection_timeout=5
            )
            return conn
        except mysql.connector.Error as err:
            if attempt == max_retries - 1:
                raise err
            print(f"⚠️ Database connection failed (attempt {attempt + 1}/{max_retries}): {err}")
            time.sleep(retry_delay)
    return None

def test_connection():
    """Test database connection"""
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return True
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
    return False