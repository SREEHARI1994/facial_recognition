import os

def remove_face_database(db_path=None):
    """
    Deletes the face database file if it exists, regardless of where this function is called from.
    Works even if called outside the installed package directory.
    """
    # If no path is provided, default to the package's installed directory
    if db_path is None:
        # Get the directory of this file (i.e., the package install folder)
        package_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(package_dir, "face_db.json")

    # Remove if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[INFO] Face database '{db_path}' removed successfully.")
    else:
        print(f"[WARN] No face database found at '{db_path}'. Nothing to remove.")
