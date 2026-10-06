"""
database.py — Async SQLAlchemy engine and session management.
Uses SQLite (via aiosqlite) for zero-config development.
Swap DATABASE_URL to PostgreSQL for production.
"""

import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# SQLite for dev; override with CLINIQ_DATABASE_URL env var for PostgreSQL
DATABASE_URL = os.environ.get(
    "CLINIQ_DATABASE_URL",
    "sqlite+aiosqlite:///./cliniq.db"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


from sqlalchemy import select

class Base(DeclarativeBase):
    pass


async def seed_default_accounts():
    """
    Seeds/updates standard default accounts for Attending Doctor and System Admin:
    - Attending Doctor: kamal@hospital.lk | Test@1234
    - System Admin:    admin@cliniq.lk   | Admin@123
    """
    from models_db import Doctor
    from auth import hash_password

    async with AsyncSessionLocal() as db:
        # 1. Attending Doctor
        res = await db.execute(select(Doctor).where(Doctor.email == "kamal@hospital.lk"))
        doc = res.scalar_one_or_none()
        if not doc:
            doc = Doctor(
                full_name="Dr. Kamal Perera",
                slmc_number="SLMC-12345",
                email="kamal@hospital.lk",
                phone="+94771234567",
                nic="198512345678",
                password_hash=hash_password("Test@1234"),
                email_verified=True,
                status="VERIFIED",
                role="DOCTOR"
            )
            db.add(doc)
            print("[SEED] Created default Attending Doctor: kamal@hospital.lk")
        else:
            doc.password_hash = hash_password("Test@1234")
            doc.status = "VERIFIED"
            doc.email_verified = True
            doc.is_suspended = False

        # 2. System Admin
        res_admin = await db.execute(select(Doctor).where(Doctor.email == "admin@cliniq.lk"))
        admin = res_admin.scalar_one_or_none()
        if not admin:
            admin = Doctor(
                full_name="Platform Administrator",
                slmc_number="SLMC-00001",
                email="admin@cliniq.lk",
                phone="+94771234567",
                nic="199012345678",
                password_hash=hash_password("Admin@123"),
                email_verified=True,
                status="VERIFIED",
                role="ADMIN"
            )
            db.add(admin)
            print("[SEED] Created default System Admin: admin@cliniq.lk")
        else:
            admin.password_hash = hash_password("Admin@123")
            admin.status = "VERIFIED"
            admin.email_verified = True
            admin.is_suspended = False

        await db.commit()


async def init_db():
    """Create all tables on startup and seed default accounts."""
    from models_db import Doctor, AuthSession, AuditLog  # noqa — import triggers registration
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_default_accounts()


async def get_db():
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

