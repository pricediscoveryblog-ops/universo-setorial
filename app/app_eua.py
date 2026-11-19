"""
Price Discovery Scanning USA — entrada do app americano no Streamlit Community Cloud.
Roda o mesmo app/app.py (mesmas abas e cálculos) no modo EUA; o app brasileiro usa app/app.py diretamente.
"""
import runpy
import sys
from pathlib import Path

_aqui = Path(__file__).resolve().parent
if str(_aqui) not in sys.path:
    sys.path.insert(0, str(_aqui))
runpy.run_path(str(_aqui / "app.py"), init_globals={"MERCADO_FIXO": "EUA"}, run_name="__main__")
