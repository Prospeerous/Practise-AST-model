"""
Migration script to add 2FA fields to the User table
Run this once to update your existing database schema
"""
from app import app, db

def add_2fa_fields():
    with app.app_context():
        try:
            # Add the new columns using raw SQL
            with db.engine.connect() as conn:
                # Check if columns already exist
                result = conn.execute(db.text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result]

                if 'totp_secret' not in columns:
                    conn.execute(db.text("ALTER TABLE users ADD COLUMN totp_secret VARCHAR(32)"))
                    conn.commit()
                    print("[OK] Added totp_secret column")
                else:
                    print("[OK] totp_secret column already exists")

                if 'is_2fa_enabled' not in columns:
                    conn.execute(db.text("ALTER TABLE users ADD COLUMN is_2fa_enabled BOOLEAN DEFAULT 0"))
                    conn.commit()
                    print("[OK] Added is_2fa_enabled column")
                else:
                    print("[OK] is_2fa_enabled column already exists")

            print("\n[OK] Database migration completed successfully!")

        except Exception as e:
            print(f"Error during migration: {e}")
            raise

if __name__ == "__main__":
    add_2fa_fields()
