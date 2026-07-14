"""
One-off data-quality check — NOT part of the package. Prints every grant
currently in the database so we can eyeball whether the scraped data is
actually correct (real grant names, sensible deadlines, non-garbage
descriptions) before building the real `report` command.

Run with: python3 inspect_data.py
"""

from accorder.storage import Grant, get_engine, session_scope

engine = get_engine()

with session_scope(engine) as session:
    grants = session.query(Grant).order_by(Grant.deadline_date).all()

print(f"Total grants in database: {len(grants)}\n")

for g in grants:
    print(f"[{g.status.value:8}] {g.grant_name}")
    print(f"           deadline: {g.deadline_date}")
    print(f"           source:   {g.source_url}")
    print(f"           desc:     {g.description[:100]}...")
    print()