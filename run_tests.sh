#!/usr/bin/env bash
# Hızlı ve Kapsamlı Test Koşucu Scripti

set -e

MODE=${1:-"quick"}

if [ "$MODE" = "quick" ] || [ "$MODE" = "fast" ]; then
    echo "⚡ [HIZLI TEST] Çekirdek modüller test ediliyor (1 saniye)..."
    .venv/bin/pytest -q tests/test_pipeline.py
elif [ "$MODE" = "all" ] || [ "$MODE" = "full" ]; then
    echo "🧪 [TAM SÜİT] 127 testin tamamı çalıştırılıyor (CI/CD modu)..."
    .venv/bin/pytest tests/
elif [ "$MODE" = "inversion" ]; then
    echo "📐 [İNVERSİYON TESTLERİ]..."
    .venv/bin/pytest -v tests/test_inversion_empirical.py
else
    echo "Kullanım: ./run_tests.sh [quick | all | inversion]"
fi
