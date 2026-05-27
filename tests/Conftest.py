# ─────────────────────────────────────────────────────────────
# tests/conftest.py
# ─────────────────────────────────────────────────────────────
# Configuration pytest partagée entre tous les fichiers de tests.
# Ajoute les dossiers services/ au path Python pour permettre
# aux fichiers de tests d'importer directement les modules
# des services sans avoir à les installer comme packages.
# ─────────────────────────────────────────────────────────────

import os
import sys

# Ajoute services/preprocessing au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "preprocessing"))

# Ajoute services/inference au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))