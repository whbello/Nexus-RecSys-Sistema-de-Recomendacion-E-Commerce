"""
catalog.py — Módulo de acceso al catálogo de productos
=======================================================
Usado por el dashboard para mostrar nombres e info enriquecida
en lugar de IDs numéricos crudos.

Generar el catálogo primero con:
    python scripts/build_product_catalog.py
"""

import json
from functools import lru_cache
from pathlib import Path

# Ruta relativa desde la raíz del proyecto
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "product_catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """Carga el catálogo una vez y lo cachea en memoria."""
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", {})


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """Retorna los metadatos del catálogo."""
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {})


def get_product(item_id: int) -> dict:
    """
    Retorna datos del producto enriquecido.
    Fallback elegante si el ítem no está en el catálogo.
    """
    catalog = load_catalog()
    item = catalog.get(str(item_id))
    if item:
        return item
    return {
        "item_id":     item_id,
        "name":        f"Producto #{item_id}",
        "category":    "Sin categoría",
        "subcategory": "General",
        "price":       None,
        "emoji":       "📦",
        "option":      "unknown",
        "description": "Producto del catálogo",
    }


def get_products_batch(item_ids: list) -> list:
    """Retorna lista de productos enriquecidos en el orden dado."""
    return [get_product(iid) for iid in item_ids]


def get_top_products(n: int = 500) -> list:
    """Retorna los n ítems de Opción A (productos reales), ordenados por popularidad."""
    catalog = load_catalog()
    opcion_a = [
        v for v in catalog.values()
        if v.get("option") == "A"
    ]
    opcion_a.sort(key=lambda x: x.get("rank_popularity") or 9999)
    return opcion_a[:n]


def catalog_available() -> bool:
    """Indica si el catálogo está disponible en disco."""
    return CATALOG_PATH.exists()
