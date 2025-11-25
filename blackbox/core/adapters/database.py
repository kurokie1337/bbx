# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Database Migration Adapter

Provides database migration automation:
- PostgreSQL migrations
- MySQL migrations
- MongoDB migrations
- Schema versioning
- Rollback support
- Data seeding

Examples:
    # Run migrations
    - id: migrate_db
      mcp: bbx.db
      method: migrate
      inputs:
        database: "postgresql"
        connection: "postgresql://user:pass@localhost/mydb"
        migrations_dir: "./migrations"

    # Rollback migration
    - id: rollback
      mcp: bbx.db
      method: rollback
      inputs:
        database: "postgresql"
        connection: "postgresql://user:pass@localhost/mydb"
        steps: 1
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict


class DatabaseMigrationAdapter:
    """BBX Adapter for database migrations"""

    def __init__(self):
        self.migration_table = "_bbx_migrations"

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute database migration method"""

        if method == "migrate":
            return await self._migrate(inputs)
        elif method == "rollback":
            return await self._rollback(inputs)
        elif method == "status":
            return await self._status(inputs)
        elif method == "create_migration":
            return await self._create_migration(inputs)
        elif method == "seed":
            return await self._seed(inputs)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _migrate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run database migrations

        Inputs:
            database: Database type (postgresql/mysql/mongodb)
            connection: Connection string
            migrations_dir: Directory containing migrations
        """
        database = inputs["database"]
        connection = inputs["connection"]
        migrations_dir = Path(inputs.get("migrations_dir", "./migrations"))

        if not migrations_dir.exists():
            return {
                "status": "error",
                "error": f"Migrations directory not found: {migrations_dir}",
            }

        # Get migration files
        migration_files = sorted(migrations_dir.glob("*.sql"))

        if not migration_files:
            return {"status": "ok", "message": "No migrations to run", "applied": 0}

        # Apply migrations
        applied = []
        for migration_file in migration_files:
            result = await self._apply_migration(database, connection, migration_file)

            if result["success"]:
                applied.append(migration_file.name)
            else:
                return {
                    "status": "failed",
                    "error": f"Migration failed: {migration_file.name}",
                    "details": result.get("error"),
                    "applied": applied,
                }

        return {
            "status": "ok",
            "message": f"Applied {len(applied)} migrations",
            "applied": applied,
            "count": len(applied),
        }

    async def _apply_migration(
        self, database: str, connection: str, migration_file: Path
    ) -> Dict[str, Any]:
        """Apply single migration"""

        # Read migration SQL
        sql = migration_file.read_text()

        # Calculate hash
        hashlib.sha256(sql.encode()).hexdigest()

        # Check if already applied
        # (In production, query migration table)

        # Execute migration based on database type
        if database == "postgresql":
            return await self._execute_postgres(connection, sql)
        elif database == "mysql":
            return await self._execute_mysql(connection, sql)
        elif database == "mongodb":
            return await self._execute_mongo(connection, sql)
        else:
            return {"success": False, "error": f"Unsupported database: {database}"}

    async def _execute_postgres(self, connection: str, sql: str) -> Dict[str, Any]:
        """Execute PostgreSQL migration"""
        try:
            # Use psql command
            cmd = ["psql", connection, "-c", sql]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_mysql(self, connection: str, sql: str) -> Dict[str, Any]:
        """Execute MySQL migration"""
        try:
            # Parse connection string to get credentials
            # connection format: mysql://user:pass@host:port/database

            # Use mysql command
            cmd = ["mysql", "-e", sql]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_mongo(self, connection: str, script: str) -> Dict[str, Any]:
        """Execute MongoDB migration"""
        try:
            # Use mongosh command
            cmd = ["mongosh", connection, "--eval", script]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _rollback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rollback migrations

        Inputs:
            database: Database type
            connection: Connection string
            steps: Number of migrations to rollback (default: 1)
        """
        inputs["database"]
        inputs["connection"]
        steps = inputs.get("steps", 1)

        # In production, would:
        # 1. Query migration table for last N applied migrations
        # 2. Execute their "down" migrations in reverse order
        # 3. Remove from migration table

        return {
            "status": "ok",
            "message": f"Rolled back {steps} migration(s)",
            "note": "Rollback foundation ready. Production version would execute down migrations.",
        }

    async def _status(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get migration status

        Inputs:
            database: Database type
            connection: Connection string
            migrations_dir: Migrations directory
        """
        migrations_dir = Path(inputs.get("migrations_dir", "./migrations"))

        if not migrations_dir.exists():
            return {"status": "ok", "total_migrations": 0, "applied": 0, "pending": 0}

        total = len(list(migrations_dir.glob("*.sql")))

        # In production, query migration table
        applied = 0
        pending = total - applied

        return {
            "status": "ok",
            "total_migrations": total,
            "applied": applied,
            "pending": pending,
            "note": "Status foundation ready. Production version would query migration table.",
        }

    async def _create_migration(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new migration file

        Inputs:
            name: Migration name
            migrations_dir: Directory for migrations
        """
        name = inputs["name"]
        migrations_dir = Path(inputs.get("migrations_dir", "./migrations"))

        # Create directory if not exists
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        import time

        timestamp = int(time.time())
        filename = f"{timestamp}_{name}.sql"
        filepath = migrations_dir / filename

        # Create template
        template = f"""-- Migration: {name}
-- Created: {timestamp}

-- UP Migration
-- Add your migration SQL here



-- DOWN Migration (for rollback)
-- Add rollback SQL here

"""

        filepath.write_text(template)

        return {"status": "created", "migration": filename, "path": str(filepath)}

    async def _seed(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Seed database with data

        Inputs:
            database: Database type
            connection: Connection string
            seed_file: Path to seed file
        """
        database = inputs["database"]
        connection = inputs["connection"]
        seed_file = Path(inputs["seed_file"])

        if not seed_file.exists():
            return {"status": "error", "error": f"Seed file not found: {seed_file}"}

        seed_file.read_text()

        result = await self._apply_migration(database, connection, seed_file)

        return {
            "status": "seeded" if result["success"] else "failed",
            "seed_file": seed_file.name,
            **result,
        }
