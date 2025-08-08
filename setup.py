"""
Setup script for HackRX LLM Query-Retrieval System
"""

import asyncio
import asyncpg
import sys
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database():
    """Setup PostgreSQL database and tables"""
    try:
        # Connect to PostgreSQL server (not specific database)
        conn = await asyncpg.connect(
            host=Config.get_database_config()["host"],
            port=Config.get_database_config()["port"],
            user=Config.get_database_config()["user"],
            password=Config.get_database_config()["password"],
            database="postgres"  # Connect to default database first
        )
        
        # Create database if it doesn't exist
        db_name = Config.get_database_config()["database"]
        
        # Check if database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        
        if not db_exists:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Created database: {db_name}")
        else:
            logger.info(f"Database {db_name} already exists")
        
        await conn.close()
        
        # Now connect to the specific database and create tables
        from database import DatabaseManager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        await db_manager.close()
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        sys.exit(1)

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "PINECONE_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == f"your-{var.lower().replace('_', '-')}":
            missing_vars.append(var)
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.error("Please set these variables before running the application")
        return False
    
    return True

async def main():
    """Main setup function"""
    logger.info("Starting HackRX LLM System Setup...")
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    # Setup database
    await setup_database()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the application with: python main.py")

if __name__ == "__main__":
    import os
    asyncio.run(main())
