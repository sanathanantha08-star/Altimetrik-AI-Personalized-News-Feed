from app.routers.feed import router as feed_router
from app.routers.preferences import router as preferences_router
from app.routers.scheduler import router as scheduler_router

__all__ = ["feed_router", "preferences_router", "scheduler_router"]
