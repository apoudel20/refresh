"""Compatibility shim: the connector lives in the standalone ``blender_mcp_connector`` package."""

from blender_mcp_connector.connector import STAGE_COLLECTION, BlenderMCPConnector, MCPConfig

__all__ = ["BlenderMCPConnector", "MCPConfig", "STAGE_COLLECTION"]
