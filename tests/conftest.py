import sys
from pathlib import Path

# Adicionar raiz do projeto ao path uma única vez
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
