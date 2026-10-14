"""One-time conversion of 1.3.5 OCR pages to project2's transcript storage."""

import zlib


def restore_document_text(connection):
    key = "project2_document_text_restored"
    if connection.execute("SELECT 1 FROM app_metadata WHERE key = ?", (key,)).fetchone():
        return
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'document_text_pages'").fetchone():
        return
    # The caller owns one transaction: corruption rolls back every converted page.
    previous_source = None
    for row in connection.execute("""SELECT s.note_id, s.id, s.name, p.data
        FROM document_text_pages p JOIN note_sources s ON s.id = p.source_id
        ORDER BY s.created_at, s.rowid, p.page_number"""):
        text = zlib.decompress(row[3]).decode("utf-8")
        prefix = "\n\n## " + row[2] + "\n\n" if row[1] != previous_source else "\n\n"
        connection.execute("UPDATE lectures SET transcript = COALESCE(transcript, '') || ? WHERE id = ?",
                           (prefix + text, row[0]))
        previous_source = row[1]
    connection.execute("INSERT INTO app_metadata(key, value) VALUES (?, 'true')", (key,))
