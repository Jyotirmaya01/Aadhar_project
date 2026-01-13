# database.py
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging
from contextlib import contextmanager
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_NAME = "aadhar_sentinel_pro.db"
DB_PATH = Path(DB_NAME)

# Connection pool settings
TIMEOUT = 30  # seconds
CHECK_SAME_THREAD = False  # Allow multi-threading


class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


@contextmanager
def get_db_connection():
    """
    Context manager for database connections
    Ensures connections are properly closed even if errors occur
    """
    conn = None
    try:
        conn = sqlite3.connect(
            DB_NAME,
            timeout=TIMEOUT,
            check_same_thread=CHECK_SAME_THREAD
        )
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        # Use WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise DatabaseError(f"Database operation failed: {e}")
    finally:
        if conn:
            conn.close()


def init_db():
    """
    Initialize database with proper schema and indexes
    
    Changes from original:
    - Added indexes for performance
    - Added constraints for data integrity
    - Added audit log table
    - Added metadata for batch tracking
    - Better column types and constraints
    """
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Main operations table with enhanced constraints
            c.execute('''
                CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operator_id TEXT NOT NULL,
                    pincode TEXT NOT NULL,
                    state TEXT NOT NULL,
                    date DATE NOT NULL,
                    enrol_adult INTEGER NOT NULL DEFAULT 0,
                    enrol_child INTEGER NOT NULL DEFAULT 0,
                    bio_update INTEGER NOT NULL DEFAULT 0,
                    demo_update INTEGER NOT NULL DEFAULT 0,
                    upload_batch_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK(enrol_adult >= 0),
                    CHECK(enrol_child >= 0),
                    CHECK(bio_update >= 0),
                    CHECK(demo_update >= 0)
                )
            ''')
            
            # Create indexes for faster queries
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_operations_date 
                ON operations(date)
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_operations_state 
                ON operations(state)
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_operations_pincode 
                ON operations(pincode)
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_operations_operator 
                ON operations(operator_id)
            ''')
            
            c.execute('''
                CREATE INDEX IF NOT EXISTS idx_operations_batch 
                ON operations(upload_batch_id)
            ''')
            
            # Batch metadata table for tracking uploads
            c.execute('''
                CREATE TABLE IF NOT EXISTS batch_metadata (
                    batch_id TEXT PRIMARY KEY,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    record_count INTEGER NOT NULL,
                    file_name TEXT,
                    status TEXT DEFAULT 'completed',
                    error_message TEXT
                )
            ''')
            
            # Audit log table for tracking changes
            c.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    record_id INTEGER,
                    user_info TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Anomaly detection results cache (optional, for performance)
            c.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_cache (
                    operation_id INTEGER PRIMARY KEY,
                    risk_score REAL,
                    risk_level TEXT,
                    deviation_score REAL,
                    is_ghost_pattern BOOLEAN,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(operation_id) REFERENCES operations(id) ON DELETE CASCADE
                )
            ''')
            
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseError(f"Failed to initialize database: {e}")


def insert_batch_data(df: pd.DataFrame, batch_id: str, file_name: str = None) -> Dict:
    """
    Bulk insert data from CSV upload with validation and transaction safety
    
    Changes from original:
    - Uses parameterized queries (prevents SQL injection)
    - Transaction management (all-or-nothing)
    - Data validation before insert
    - Batch metadata tracking
    - Returns detailed status
    
    Args:
        df: DataFrame containing the data to insert
        batch_id: Unique identifier for this batch
        file_name: Original filename (optional)
    
    Returns:
        Dictionary with status information
    """
    try:
        # Validate DataFrame
        if df.empty:
            raise ValueError("Cannot insert empty DataFrame")
        
        # Standardize column names
        rename_map = {
            'OperatorID': 'operator_id',
            'Pincode': 'pincode',
            'State': 'state',
            'Date': 'date',
            'Adult_Enrolment': 'enrol_adult',
            'Child_Enrolment': 'enrol_child',
            'Bio_Update': 'bio_update',
            'Demo_Update': 'demo_update'
        }
        df = df.rename(columns=rename_map)
        
        # Validate required columns exist
        required_cols = ['operator_id', 'pincode', 'state', 'date', 
                        'enrol_adult', 'enrol_child', 'bio_update', 'demo_update']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add batch_id column
        df['upload_batch_id'] = batch_id
        
        # Clean and validate data
        # Ensure numeric columns are integers
        numeric_cols = ['enrol_adult', 'enrol_child', 'bio_update', 'demo_update']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            # Ensure non-negative
            df[col] = df[col].clip(lower=0)
        
        # Sanitize string columns (prevent SQL injection)
        string_cols = ['operator_id', 'pincode', 'state']
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            # Remove potentially dangerous characters
            df[col] = df[col].str.replace(r'[;\'"\\]', '', regex=True)
            # Limit length
            df[col] = df[col].str[:100]
        
        # Validate and format date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid dates, removing those records")
            df = df[df['date'].notna()]
        
        if df.empty:
            raise ValueError("No valid records remaining after validation")
        
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Select final columns
        insert_cols = required_cols + ['upload_batch_id']
        df_insert = df[insert_cols].copy()
        
        record_count = len(df_insert)
        
        with get_db_connection() as conn:
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Insert data using parameterized query (SQL injection safe)
                df_insert.to_sql(
                    'operations',
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi'  # Faster batch insert
                )
                
                # Insert batch metadata
                conn.execute('''
                    INSERT INTO batch_metadata (batch_id, record_count, file_name, status)
                    VALUES (?, ?, ?, ?)
                ''', (batch_id, record_count, file_name, 'completed'))
                
                # Log the action
                conn.execute('''
                    INSERT INTO audit_log (action, table_name, details)
                    VALUES (?, ?, ?)
                ''', ('INSERT_BATCH', 'operations', 
                      f'Batch {batch_id}: {record_count} records'))
                
                conn.commit()
                
                logger.info(f"Successfully inserted {record_count} records for batch {batch_id}")
                
                return {
                    'success': True,
                    'batch_id': batch_id,
                    'records_inserted': record_count,
                    'message': f'Successfully inserted {record_count} records'
                }
                
            except Exception as e:
                conn.rollback()
                
                # Log failed batch
                conn.execute('''
                    INSERT INTO batch_metadata (batch_id, record_count, file_name, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (batch_id, 0, file_name, 'failed', str(e)))
                conn.commit()
                
                raise
                
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        raise DatabaseError(f"Failed to insert batch data: {e}")


def fetch_time_series_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    state: Optional[str] = None,
    operator_id: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch time series data with optional filtering
    
    Changes from original:
    - Parameterized queries (SQL injection safe)
    - Optional date range filtering
    - Optional state/operator filtering
    - Limit support for large datasets
    - Better error handling
    
    Args:
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        state: Filter by state
        operator_id: Filter by operator
        limit: Maximum records to return
    
    Returns:
        DataFrame with operations data
    """
    try:
        # Build query with parameterized filters
        query = "SELECT * FROM operations WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        if state:
            query += " AND state = ?"
            params.append(state)
        
        if operator_id:
            query += " AND operator_id = ?"
            params.append(operator_id)
        
        query += " ORDER BY date ASC, state ASC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
            # Convert date column to datetime
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Fetched {len(df)} records from database")
            return df
            
    except Exception as e:
        logger.error(f"Failed to fetch time series data: {e}")
        raise DatabaseError(f"Failed to fetch data: {e}")


def get_batch_info(batch_id: str) -> Optional[Dict]:
    """
    Get information about a specific batch
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        Dictionary with batch information or None if not found
    """
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT batch_id, upload_timestamp, record_count, file_name, status, error_message
                FROM batch_metadata
                WHERE batch_id = ?
            ''', (batch_id,))
            
            row = c.fetchone()
            if row:
                return {
                    'batch_id': row[0],
                    'upload_timestamp': row[1],
                    'record_count': row[2],
                    'file_name': row[3],
                    'status': row[4],
                    'error_message': row[5]
                }
            return None
            
    except Exception as e:
        logger.error(f"Failed to get batch info: {e}")
        return None


def get_all_batches() -> List[Dict]:
    """
    Get information about all uploaded batches
    
    Returns:
        List of dictionaries with batch information
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query('''
                SELECT batch_id, upload_timestamp, record_count, file_name, status
                FROM batch_metadata
                ORDER BY upload_timestamp DESC
            ''', conn)
            
            return df.to_dict('records')
            
    except Exception as e:
        logger.error(f"Failed to get batches: {e}")
        return []


def delete_batch(batch_id: str) -> bool:
    """
    Delete a batch and all associated records
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Delete operations
                c.execute('DELETE FROM operations WHERE upload_batch_id = ?', (batch_id,))
                deleted_count = c.rowcount
                
                # Delete batch metadata
                c.execute('DELETE FROM batch_metadata WHERE batch_id = ?', (batch_id,))
                
                # Log action
                c.execute('''
                    INSERT INTO audit_log (action, table_name, details)
                    VALUES (?, ?, ?)
                ''', ('DELETE_BATCH', 'operations', 
                      f'Deleted batch {batch_id}: {deleted_count} records'))
                
                conn.commit()
                logger.info(f"Deleted batch {batch_id} with {deleted_count} records")
                return True
                
            except Exception as e:
                conn.rollback()
                raise
                
    except Exception as e:
        logger.error(f"Failed to delete batch: {e}")
        return False


def get_database_stats() -> Dict:
    """
    Get overall database statistics
    
    Returns:
        Dictionary with database statistics
    """
    try:
        with get_db_connection() as conn:
            stats = {}
            
            # Total records
            stats['total_records'] = pd.read_sql_query(
                'SELECT COUNT(*) as count FROM operations', conn
            )['count'].iloc[0]
            
            # Total batches
            stats['total_batches'] = pd.read_sql_query(
                'SELECT COUNT(*) as count FROM batch_metadata', conn
            )['count'].iloc[0]
            
            # Date range
            date_range = pd.read_sql_query('''
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM operations
            ''', conn)
            stats['date_range'] = {
                'start': date_range['min_date'].iloc[0],
                'end': date_range['max_date'].iloc[0]
            }
            
            # Unique states
            stats['unique_states'] = pd.read_sql_query(
                'SELECT COUNT(DISTINCT state) as count FROM operations', conn
            )['count'].iloc[0]
            
            # Unique operators
            stats['unique_operators'] = pd.read_sql_query(
                'SELECT COUNT(DISTINCT operator_id) as count FROM operations', conn
            )['count'].iloc[0]
            
            # Database size
            if DB_PATH.exists():
                stats['db_size_mb'] = DB_PATH.stat().st_size / (1024 * 1024)
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


def cache_anomaly_results(operation_id: int, risk_score: float, 
                          risk_level: str, deviation_score: float,
                          is_ghost_pattern: bool) -> bool:
    """
    Cache anomaly detection results for performance
    
    Args:
        operation_id: ID of the operation record
        risk_score: Calculated risk score
        risk_level: Risk level (Low/Medium/High)
        deviation_score: Deviation from baseline
        is_ghost_pattern: Whether ghost pattern detected
    
    Returns:
        True if successful
    """
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO anomaly_cache 
                (operation_id, risk_score, risk_level, deviation_score, is_ghost_pattern)
                VALUES (?, ?, ?, ?, ?)
            ''', (operation_id, risk_score, risk_level, deviation_score, is_ghost_pattern))
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to cache anomaly results: {e}")
        return False


def get_cached_anomalies(min_risk_score: float = 70) -> pd.DataFrame:
    """
    Retrieve cached anomaly results
    
    Args:
        min_risk_score: Minimum risk score to filter
    
    Returns:
        DataFrame with cached anomaly data
    """
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT o.*, a.risk_score, a.risk_level, a.deviation_score, 
                       a.is_ghost_pattern, a.calculated_at
                FROM operations o
                INNER JOIN anomaly_cache a ON o.id = a.operation_id
                WHERE a.risk_score >= ?
                ORDER BY a.risk_score DESC
            '''
            df = pd.read_sql_query(query, conn, params=(min_risk_score,))
            return df
            
    except Exception as e:
        logger.error(f"Failed to get cached anomalies: {e}")
        return pd.DataFrame()


def backup_database(backup_path: str = None) -> bool:
    """
    Create a backup of the database
    
    Args:
        backup_path: Path for backup file (optional)
    
    Returns:
        True if successful
    """
    try:
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"aadhar_sentinel_backup_{timestamp}.db"
        
        import shutil
        shutil.copy2(DB_NAME, backed_path)

        logger.info(f"Database backed up to {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False


# Initialize database on module import
if __name__ != "__main__":
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database on import: {e}")

