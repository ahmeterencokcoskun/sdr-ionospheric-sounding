#!/usr/bin/env bash
set -e

echo "⚡ [TEST] IONO-SENSE TA7CAE birim ve entegrasyon testleri çalıştırılıyor..."
.venv/bin/pytest -v tests/test_pipeline.py
