"""sql_query tool: read-only SELECT queries against a demo SQLite database.

Read-only is enforced three ways (defense in depth):
1. The query must be a single statement starting with SELECT or WITH.
2. The database is opened in SQLite's read-only mode (mode=ro).
3. An authorizer callback denies anything other than reading.

The demo data (employees, products, sales) is generated with a fixed random
seed, so results are the same on every machine and evals can check exact values.
"""

from __future__ import annotations

import random
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from langchain_core.tools import tool

from agentlite.config import get_settings

MAX_ROWS = 50

SCHEMA = """employees(id, name, department, title, salary, hire_date, city)
products(id, name, category, price)
sales(id, product_id -> products.id, employee_id -> employees.id, quantity, sale_date, region)"""

ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def _authorizer(action: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def check_query(query: str) -> str | None:
    """Return an error message if the query isn't a single read-only SELECT."""
    stripped = query.strip().rstrip(";").strip()
    if ";" in stripped:
        return "only one statement is allowed"
    if not re.match(r"(?is)^\s*(select|with)\b", stripped):
        return "only SELECT queries are allowed"
    return None


@tool(parse_docstring=True)
def sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query on the company database (SQLite) and return the rows.

    Tables:
    employees(id, name, department, title, salary, hire_date, city)
    products(id, name, category, price)
    sales(id, product_id, employee_id, quantity, sale_date, region)
    Revenue for a sale is sales.quantity * products.price. Dates are 'YYYY-MM-DD' text.

    Args:
        query: One SQLite SELECT statement, e.g. "SELECT department, COUNT(*) FROM employees GROUP BY department".
    """
    problem = check_query(query)
    if problem:
        return f"ERROR: {problem}. Tables: {SCHEMA}"

    db_path = ensure_demo_db()
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        conn.set_authorizer(_authorizer)
        try:
            cursor = conn.execute(query.strip().rstrip(";"))
            columns = [c[0] for c in cursor.description or []]
            rows = cursor.fetchmany(MAX_ROWS + 1)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"ERROR: SQL error: {exc}. Tables: {SCHEMA}"

    if not rows:
        return "Query returned no rows."
    lines = [" | ".join(columns)]
    lines += [" | ".join(str(v) for v in row) for row in rows[:MAX_ROWS]]
    if len(rows) > MAX_ROWS:
        lines.append(f"...(more than {MAX_ROWS} rows; add LIMIT or aggregate)")
    return "\n".join(lines)


# --- Demo database -------------------------------------------------------------

DEPARTMENTS = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Data Engineer"],
    "Sales": ["Account Executive", "Sales Manager"],
    "Marketing": ["Marketing Specialist", "Content Lead"],
    "Support": ["Support Agent", "Support Lead"],
}
FIRST = ["Ava", "Ben", "Chloe", "Dev", "Ella", "Finn", "Grace", "Hugo", "Isla", "Jay", "Kira", "Leo",
         "Maya", "Nate", "Olivia", "Priya", "Quinn", "Ravi", "Sara", "Tom", "Uma", "Vik", "Wren", "Zoe"]
LAST = ["Patel", "Nguyen", "Smith", "Garcia", "Kim", "Okafor", "Rossi", "Chen", "Silva", "Brown"]
CITIES = ["London", "Toronto", "Austin", "Berlin", "Bangalore"]
PRODUCTS = [
    ("Laptop Pro 14", "Hardware", 1499.0), ("Laptop Air 13", "Hardware", 999.0),
    ("4K Monitor", "Hardware", 399.0), ("Wireless Mouse", "Accessories", 29.0),
    ("Mechanical Keyboard", "Accessories", 89.0), ("USB-C Dock", "Accessories", 149.0),
    ("Cloud Backup (1 yr)", "Software", 59.0), ("Office Suite (1 yr)", "Software", 99.0),
    ("Antivirus (1 yr)", "Software", 39.0), ("Support Plan", "Services", 199.0),
]
REGIONS = ["North America", "Europe", "Asia"]


def seed_demo_db(path: Path) -> None:
    """Create the demo database from scratch with deterministic data."""
    rng = random.Random(42)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, title TEXT,
                                salary INTEGER, hire_date TEXT, city TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE sales (id INTEGER PRIMARY KEY, product_id INTEGER REFERENCES products(id),
                            employee_id INTEGER REFERENCES employees(id), quantity INTEGER,
                            sale_date TEXT, region TEXT);
        """
    )
    employees = []
    for i in range(1, 41):
        dept = rng.choice(list(DEPARTMENTS))
        employees.append((
            i, f"{rng.choice(FIRST)} {rng.choice(LAST)}", dept, rng.choice(DEPARTMENTS[dept]),
            rng.randrange(45_000, 160_000, 1000),
            (date(2016, 1, 1) + timedelta(days=rng.randrange(0, 3000))).isoformat(), rng.choice(CITIES),
        ))
    conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [(i, *p) for i, p in enumerate(PRODUCTS, 1)])

    sales_staff = [e[0] for e in employees if e[2] == "Sales"]
    sales = [
        (i, rng.randint(1, len(PRODUCTS)), rng.choice(sales_staff), rng.randint(1, 10),
         (date(2024, 1, 1) + timedelta(days=rng.randrange(0, 366))).isoformat(), rng.choice(REGIONS))
        for i in range(1, 501)
    ]
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?)", sales)
    conn.commit()
    conn.close()


def ensure_demo_db() -> Path:
    path = get_settings().demo_db_path
    if not path.exists():
        seed_demo_db(path)
    return path
