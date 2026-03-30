#!/bin/bash
# Script de lancement automatique — Veille Éditoriale
cd "/Users/romainsaillet/Documents/Veille Edito"
ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)
export ANTHROPIC_API_KEY
/Library/Developer/CommandLineTools/usr/bin/python3 veille.py >> logs/veille.log 2>&1
