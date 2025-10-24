"""Paquete `medical`.

Nota: no importamos submódulos en el nivel del paquete para evitar
importaciones circulares (por ejemplo `medical.cli` -> `medical.system`).
Importa explícitamente `medical.cli` o `medical.system` desde el código
que los vaya a usar (p. ej. `app.py`).
"""

__all__ = []
