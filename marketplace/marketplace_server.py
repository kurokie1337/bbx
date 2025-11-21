"""
BBX App Marketplace Server
Full-featured marketplace for discovering, publishing, and managing BBX workflows
"""
import asyncio
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import aiofiles


app = FastAPI(title="BBX Marketplace", version="1.0.0")

# Database setup
DB_PATH = Path("marketplace.db")
PACKAGES_DIR = Path("packages")
PACKAGES_DIR.mkdir(exist_ok=True)


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize marketplace database"""
    conn = get_db()
    cursor = conn.cursor()

    # Apps table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            package_name TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            description TEXT,
            author TEXT NOT NULL,
            author_email TEXT,
            license TEXT DEFAULT 'Apache-2.0',
            category TEXT,
            tags TEXT,
            min_bbx_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            downloads INTEGER DEFAULT 0,
            rating REAL DEFAULT 0.0,
            rating_count INTEGER DEFAULT 0,
            featured BOOLEAN DEFAULT 0,
            verified BOOLEAN DEFAULT 0,
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_hash TEXT NOT NULL
        )
    """)

    # Reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (app_id) REFERENCES apps(id)
        )
    """)

    # Dependencies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dependencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_id INTEGER NOT NULL,
            dependency_package TEXT NOT NULL,
            version_constraint TEXT,
            FOREIGN KEY (app_id) REFERENCES apps(id)
        )
    """)

    # Categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            icon TEXT
        )
    """)

    # Insert default categories
    default_categories = [
        ("Development", "Development tools and utilities", "🛠️"),
        ("DevOps", "CI/CD and infrastructure automation", "🚀"),
        ("Data", "Data processing and analytics", "📊"),
        ("AI/ML", "Artificial Intelligence and Machine Learning", "🤖"),
        ("Cloud", "Cloud platform integrations", "☁️"),
        ("Security", "Security and compliance tools", "🔒"),
        ("Monitoring", "Monitoring and observability", "📈"),
        ("Utilities", "General utilities and helpers", "🔧"),
    ]

    for name, desc, icon in default_categories:
        cursor.execute(
            "INSERT OR IGNORE INTO categories (name, description, icon) VALUES (?, ?, ?)",
            (name, desc, icon)
        )

    conn.commit()
    conn.close()


# Pydantic models
class AppMetadata(BaseModel):
    package_name: str = Field(..., pattern=r"^[a-z0-9\-_]+$")
    name: str
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: Optional[str] = None
    author: str
    author_email: Optional[str] = None
    license: str = "Apache-2.0"
    category: Optional[str] = None
    tags: List[str] = []
    min_bbx_version: Optional[str] = None
    dependencies: List[Dict[str, str]] = []


class ReviewCreate(BaseModel):
    user_name: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class SearchQuery(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = []
    min_rating: Optional[float] = None
    featured_only: bool = False
    verified_only: bool = False
    limit: int = 20
    offset: int = 0


# API Endpoints

@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Marketplace API info"""
    return {
        "name": "BBX App Marketplace",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "search": "/apps/search",
            "publish": "/apps/publish",
            "download": "/apps/{package_name}/download",
            "categories": "/categories"
        }
    }


@app.post("/apps/publish")
async def publish_app(
    metadata: AppMetadata,
    file: UploadFile = File(...)
):
    """Publish a new app or update existing one"""

    # Validate file is .bbx
    if not file.filename.endswith(".bbx"):
        raise HTTPException(400, "Only .bbx files are allowed")

    # Read and hash file
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    file_size = len(content)

    # Save file
    file_path = PACKAGES_DIR / f"{metadata.package_name}-{metadata.version}.bbx"
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Insert into database
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO apps (
                package_name, name, version, description, author, author_email,
                license, category, tags, min_bbx_version, file_path, file_size, file_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.package_name,
            metadata.name,
            metadata.version,
            metadata.description,
            metadata.author,
            metadata.author_email,
            metadata.license,
            metadata.category,
            ",".join(metadata.tags),
            metadata.min_bbx_version,
            str(file_path),
            file_size,
            file_hash
        ))

        app_id = cursor.lastrowid

        # Insert dependencies
        for dep in metadata.dependencies:
            cursor.execute("""
                INSERT INTO dependencies (app_id, dependency_package, version_constraint)
                VALUES (?, ?, ?)
            """, (app_id, dep["package"], dep.get("version", "*")))

        conn.commit()

        return {
            "status": "published",
            "app_id": app_id,
            "package_name": metadata.package_name,
            "version": metadata.version,
            "hash": file_hash
        }

    except sqlite3.IntegrityError:
        conn.rollback()
        raise HTTPException(409, "App with this package name and version already exists")
    finally:
        conn.close()


@app.post("/apps/search")
async def search_apps(query: SearchQuery):
    """Search for apps in the marketplace"""
    conn = get_db()
    cursor = conn.cursor()

    sql = "SELECT * FROM apps WHERE 1=1"
    params = []

    if query.query:
        sql += " AND (name LIKE ? OR description LIKE ? OR tags LIKE ?)"
        search_term = f"%{query.query}%"
        params.extend([search_term, search_term, search_term])

    if query.category:
        sql += " AND category = ?"
        params.append(query.category)

    if query.tags:
        for tag in query.tags:
            sql += " AND tags LIKE ?"
            params.append(f"%{tag}%")

    if query.min_rating:
        sql += " AND rating >= ?"
        params.append(query.min_rating)

    if query.featured_only:
        sql += " AND featured = 1"

    if query.verified_only:
        sql += " AND verified = 1"

    sql += " ORDER BY downloads DESC, rating DESC LIMIT ? OFFSET ?"
    params.extend([query.limit, query.offset])

    cursor.execute(sql, params)
    rows = cursor.fetchall()

    apps = []
    for row in rows:
        apps.append(dict(row))

    conn.close()

    return {
        "results": apps,
        "count": len(apps),
        "limit": query.limit,
        "offset": query.offset
    }


@app.get("/apps/{package_name}")
async def get_app_details(package_name: str):
    """Get detailed information about an app"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM apps WHERE package_name = ? ORDER BY created_at DESC LIMIT 1", (package_name,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "App not found")

    app = dict(row)

    # Get dependencies
    cursor.execute("SELECT * FROM dependencies WHERE app_id = ?", (app["id"],))
    dependencies = [dict(dep) for dep in cursor.fetchall()]
    app["dependencies"] = dependencies

    # Get reviews
    cursor.execute("SELECT * FROM reviews WHERE app_id = ? ORDER BY created_at DESC LIMIT 10", (app["id"],))
    reviews = [dict(review) for review in cursor.fetchall()]
    app["reviews"] = reviews

    conn.close()

    return app


@app.get("/apps/{package_name}/download")
async def download_app(package_name: str, version: Optional[str] = None):
    """Download an app package"""
    conn = get_db()
    cursor = conn.cursor()

    if version:
        cursor.execute(
            "SELECT * FROM apps WHERE package_name = ? AND version = ?",
            (package_name, version)
        )
    else:
        cursor.execute(
            "SELECT * FROM apps WHERE package_name = ? ORDER BY created_at DESC LIMIT 1",
            (package_name,)
        )

    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "App not found")

    app = dict(row)

    # Increment download count
    cursor.execute("UPDATE apps SET downloads = downloads + 1 WHERE id = ?", (app["id"],))
    conn.commit()
    conn.close()

    file_path = Path(app["file_path"])
    if not file_path.exists():
        raise HTTPException(404, "App file not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=f"{package_name}-{app['version']}.bbx"
    )


@app.post("/apps/{package_name}/review")
async def add_review(package_name: str, review: ReviewCreate):
    """Add a review for an app"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM apps WHERE package_name = ? ORDER BY created_at DESC LIMIT 1", (package_name,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "App not found")

    app_id = row[0]

    cursor.execute("""
        INSERT INTO reviews (app_id, user_name, rating, comment)
        VALUES (?, ?, ?, ?)
    """, (app_id, review.user_name, review.rating, review.comment))

    # Update app rating
    cursor.execute("""
        UPDATE apps SET
            rating = (SELECT AVG(rating) FROM reviews WHERE app_id = ?),
            rating_count = (SELECT COUNT(*) FROM reviews WHERE app_id = ?)
        WHERE id = ?
    """, (app_id, app_id, app_id))

    conn.commit()
    conn.close()

    return {"status": "review_added"}


@app.get("/categories")
async def get_categories():
    """Get all categories"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM categories ORDER BY name")
    categories = [dict(row) for row in cursor.fetchall()]

    # Get app counts for each category
    for cat in categories:
        cursor.execute("SELECT COUNT(*) as count FROM apps WHERE category = ?", (cat["name"],))
        cat["app_count"] = cursor.fetchone()[0]

    conn.close()

    return {"categories": categories}


@app.get("/featured")
async def get_featured_apps(limit: int = 10):
    """Get featured apps"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM apps WHERE featured = 1
        ORDER BY rating DESC, downloads DESC
        LIMIT ?
    """, (limit,))

    apps = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return {"featured": apps}


@app.get("/stats")
async def get_marketplace_stats():
    """Get marketplace statistics"""
    conn = get_db()
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) as count FROM apps")
    stats["total_apps"] = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(downloads) as total FROM apps")
    stats["total_downloads"] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) as count FROM reviews")
    stats["total_reviews"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(rating) as avg FROM apps WHERE rating > 0")
    stats["average_rating"] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(DISTINCT author) as count FROM apps")
    stats["total_authors"] = cursor.fetchone()[0]

    conn.close()

    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
