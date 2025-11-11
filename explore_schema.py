#!/usr/bin/env python
"""
Explore the database schema to document all tables and columns
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "nfl_merged.db"

def explore_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print("="*80)
    print("NFL DATABASE SCHEMA EXPLORATION")
    print("="*80)

    schema_info = {}

    for table in tables:
        print(f"\n\n{'='*80}")
        print(f"TABLE: {table}")
        print(f"{'='*80}")

        # Get table info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()

        print(f"\nColumns ({len(columns)} total):")
        print("-"*80)

        column_info = []
        for col in columns:
            col_id, col_name, col_type, not_null, default_val, is_pk = col
            pk_marker = " [PRIMARY KEY]" if is_pk else ""
            nn_marker = " [NOT NULL]" if not_null else ""
            print(f"  {col_name:<40} {col_type:<15} {pk_marker}{nn_marker}")
            column_info.append({
                'name': col_name,
                'type': col_type,
                'primary_key': bool(is_pk),
                'not_null': bool(not_null)
            })

        # Get sample data
        try:
            sample = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3", conn)
            print(f"\nSample Data (first 3 rows):")
            print("-"*80)
            print(sample.to_string())

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            print(f"\nTotal Rows: {row_count:,}")
        except Exception as e:
            print(f"\nError fetching sample data: {e}")

        schema_info[table] = {
            'columns': column_info,
            'row_count': row_count if 'row_count' in locals() else 0
        }

    conn.close()

    # Save to file
    output_file = Path(__file__).parent / "schema_exploration.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("NFL DATABASE SCHEMA\n")
        f.write("="*80 + "\n\n")

        for table, info in schema_info.items():
            f.write(f"\nTABLE: {table}\n")
            f.write("-"*80 + "\n")
            f.write(f"Rows: {info['row_count']:,}\n\n")
            f.write("Columns:\n")
            for col in info['columns']:
                pk = " [PK]" if col['primary_key'] else ""
                nn = " [NN]" if col['not_null'] else ""
                f.write(f"  {col['name']:<40} {col['type']:<15}{pk}{nn}\n")
            f.write("\n")

    print(f"\n\n{'='*80}")
    print(f"Schema information saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    explore_schema()
