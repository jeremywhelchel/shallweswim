"""API package - re-exports from routes module."""

from shallweswim.api.routes import initialize_location_data, register_routes

__all__ = ["initialize_location_data", "register_routes"]
