# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BBX Marketplace Client
Command-line client for interacting with the BBX App Marketplace
"""
import asyncio
import json
from pathlib import Path
from typing import Optional, List
import aiohttp
import click


class MarketplaceClient:
    """Client for BBX Marketplace API"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: List[str] = None,
        featured: bool = False,
        verified: bool = False,
        limit: int = 20
    ):
        """Search for apps"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": query,
                "category": category,
                "tags": tags or [],
                "featured_only": featured,
                "verified_only": verified,
                "limit": limit,
                "offset": 0
            }

            async with session.post(f"{self.base_url}/apps/search", json=payload) as resp:
                return await resp.json()

    async def get_app(self, package_name: str):
        """Get app details"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/apps/{package_name}") as resp:
                if resp.status == 404:
                    return None
                return await resp.json()

    async def download(self, package_name: str, output: Path, version: Optional[str] = None):
        """Download an app"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/apps/{package_name}/download"
            if version:
                url += f"?version={version}"

            async with session.get(url) as resp:
                if resp.status == 404:
                    return False

                content = await resp.read()
                output.write_bytes(content)
                return True

    async def publish(self, metadata_file: Path, bbx_file: Path):
        """Publish an app"""
        metadata = json.loads(metadata_file.read_text())

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()

            # Add metadata fields
            for key, value in metadata.items():
                if isinstance(value, list):
                    form.add_field(key, json.dumps(value))
                else:
                    form.add_field(key, str(value))

            # Add file
            form.add_field(
                "file",
                bbx_file.read_bytes(),
                filename=bbx_file.name,
                content_type="application/octet-stream"
            )

            async with session.post(f"{self.base_url}/apps/publish", data=form) as resp:
                return await resp.json()

    async def add_review(self, package_name: str, rating: int, user_name: str, comment: Optional[str] = None):
        """Add a review"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "user_name": user_name,
                "rating": rating,
                "comment": comment
            }

            async with session.post(f"{self.base_url}/apps/{package_name}/review", json=payload) as resp:
                return await resp.json()

    async def get_categories(self):
        """Get all categories"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/categories") as resp:
                return await resp.json()

    async def get_featured(self, limit: int = 10):
        """Get featured apps"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/featured?limit={limit}") as resp:
                return await resp.json()

    async def get_stats(self):
        """Get marketplace statistics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/stats") as resp:
                return await resp.json()


# CLI Commands

@click.group()
@click.option("--server", default="http://localhost:8080", help="Marketplace server URL")
@click.pass_context
def cli(ctx, server):
    """BBX Marketplace CLI"""
    ctx.obj = MarketplaceClient(server)


@cli.command()
@click.argument("query", required=False)
@click.option("--category", help="Filter by category")
@click.option("--tag", multiple=True, help="Filter by tags")
@click.option("--featured", is_flag=True, help="Show only featured apps")
@click.option("--verified", is_flag=True, help="Show only verified apps")
@click.option("--limit", default=20, help="Number of results")
@click.pass_obj
def search(client, query, category, tag, featured, verified, limit):
    """Search for apps in the marketplace"""
    result = asyncio.run(
        client.search(
            query=query,
            category=category,
            tags=list(tag),
            featured=featured,
            verified=verified,
            limit=limit
        )
    )

    if not result["results"]:
        click.echo("No apps found")
        return

    click.echo(f"\nFound {result['count']} apps:\n")

    for app in result["results"]:
        status = []
        if app.get("featured"):
            status.append("⭐ Featured")
        if app.get("verified"):
            status.append("✓ Verified")

        click.echo(f"{app['name']} ({app['package_name']})")
        click.echo(f"  Version: {app['version']}")
        click.echo(f"  Author: {app['author']}")
        if app.get("description"):
            click.echo(f"  Description: {app['description']}")
        click.echo(f"  Rating: {app['rating']:.1f} ⭐ ({app['rating_count']} reviews)")
        click.echo(f"  Downloads: {app['downloads']}")
        if status:
            click.echo(f"  Status: {' | '.join(status)}")
        click.echo()


@cli.command()
@click.argument("package_name")
@click.pass_obj
def info(client, package_name):
    """Get detailed information about an app"""
    app = asyncio.run(client.get_app(package_name))

    if not app:
        click.echo(f"App not found: {package_name}")
        return

    click.echo(f"\n{app['name']}")
    click.echo("=" * len(app['name']))
    click.echo(f"Package: {app['package_name']}")
    click.echo(f"Version: {app['version']}")
    click.echo(f"Author: {app['author']}")
    if app.get("author_email"):
        click.echo(f"Email: {app['author_email']}")
    click.echo(f"License: {app['license']}")
    if app.get("category"):
        click.echo(f"Category: {app['category']}")
    if app.get("tags"):
        click.echo(f"Tags: {app['tags']}")
    click.echo(f"\nDescription:\n{app.get('description', 'No description')}")
    click.echo(f"\nRating: {app['rating']:.1f} ⭐ ({app['rating_count']} reviews)")
    click.echo(f"Downloads: {app['downloads']}")

    if app.get("dependencies"):
        click.echo("\nDependencies:")
        for dep in app["dependencies"]:
            version = dep.get("version_constraint", "*")
            click.echo(f"  - {dep['dependency_package']} ({version})")

    if app.get("reviews"):
        click.echo("\nRecent Reviews:")
        for review in app["reviews"][:5]:
            click.echo(f"\n  {review['user_name']} - {review['rating']} ⭐")
            if review.get("comment"):
                click.echo(f"  {review['comment']}")


@cli.command()
@click.argument("package_name")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--version", help="Specific version to download")
@click.pass_obj
def install(client, package_name, output, version):
    """Download and install an app"""
    if not output:
        output = f"{package_name}.bbx"

    output_path = Path(output)

    click.echo(f"Downloading {package_name}...")

    success = asyncio.run(client.download(package_name, output_path, version))

    if not success:
        click.echo(f"Failed to download {package_name}")
        return

    click.echo(f"Downloaded to {output_path}")
    click.echo(f"Size: {output_path.stat().st_size} bytes")


@cli.command()
@click.argument("metadata_file", type=click.Path(exists=True))
@click.argument("bbx_file", type=click.Path(exists=True))
@click.pass_obj
def publish(client, metadata_file, bbx_file):
    """Publish an app to the marketplace"""
    metadata_path = Path(metadata_file)
    bbx_path = Path(bbx_file)

    click.echo(f"Publishing {bbx_path.name}...")

    result = asyncio.run(client.publish(metadata_path, bbx_path))

    if "error" in result:
        click.echo(f"Failed to publish: {result['error']}")
        return

    click.echo("Successfully published!")
    click.echo(f"App ID: {result['app_id']}")
    click.echo(f"Package: {result['package_name']}")
    click.echo(f"Version: {result['version']}")
    click.echo(f"Hash: {result['hash']}")


@cli.command()
@click.argument("package_name")
@click.argument("rating", type=click.IntRange(1, 5))
@click.option("--user", default="anonymous", help="Your username")
@click.option("--comment", help="Review comment")
@click.pass_obj
def review(client, package_name, rating, user, comment):
    """Add a review for an app"""
    result = asyncio.run(client.add_review(package_name, rating, user, comment))

    if "error" in result:
        click.echo(f"Failed to add review: {result['error']}")
        return

    click.echo("Review added successfully!")


@cli.command()
@click.pass_obj
def categories(client):
    """List all categories"""
    result = asyncio.run(client.get_categories())

    click.echo("\nCategories:\n")

    for cat in result["categories"]:
        icon = cat.get("icon", "")
        click.echo(f"{icon} {cat['name']} ({cat['app_count']} apps)")
        if cat.get("description"):
            click.echo(f"   {cat['description']}")
        click.echo()


@cli.command()
@click.option("--limit", default=10, help="Number of apps to show")
@click.pass_obj
def featured(client, limit):
    """Show featured apps"""
    result = asyncio.run(client.get_featured(limit))

    click.echo("\n⭐ Featured Apps:\n")

    for app in result["featured"]:
        click.echo(f"{app['name']} ({app['package_name']})")
        click.echo(f"  {app.get('description', '')}")
        click.echo(f"  Rating: {app['rating']:.1f} ⭐ | Downloads: {app['downloads']}")
        click.echo()


@cli.command()
@click.pass_obj
def stats(client):
    """Show marketplace statistics"""
    result = asyncio.run(client.get_stats())

    click.echo("\nMarketplace Statistics:\n")
    click.echo(f"Total Apps: {result['total_apps']}")
    click.echo(f"Total Downloads: {result['total_downloads']}")
    click.echo(f"Total Reviews: {result['total_reviews']}")
    click.echo(f"Average Rating: {result['average_rating']:.2f} ⭐")
    click.echo(f"Total Authors: {result['total_authors']}")


if __name__ == "__main__":
    cli()
