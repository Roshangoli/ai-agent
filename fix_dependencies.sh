#!/bin/bash

# Fix Dependencies Script for AI Multi-Agent Analytics System

echo "🔧 Fixing Dependencies for AI Analytics System"
echo "=============================================="

# Check Python version
echo ""
echo "📋 Checking Python version..."
python --version

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
    echo "   Recommendation: Create one with 'python -m venv venv'"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Backup current requirements
echo ""
echo "💾 Backing up current environment..."
pip freeze > requirements_backup.txt
echo "✅ Saved to requirements_backup.txt"

# Option 1: Minimal fix (just update OpenAI)
echo ""
echo "🔧 Fix Option 1: Update OpenAI for AutoGen"
echo "   pip install --upgrade 'openai>=1.66.2'"
read -p "   Apply this fix? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install --upgrade 'openai>=1.66.2'
    echo "✅ OpenAI updated"
fi

# Option 2: Fix LangChain compatibility
echo ""
echo "🔧 Fix Option 2: Fix LangChain Pydantic compatibility"
echo "   pip install 'pydantic<2.0' --force-reinstall"
read -p "   Apply this fix? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install 'pydantic<2.0' --force-reinstall
    echo "✅ Pydantic downgraded"
fi

# Option 3: Install all requirements fresh
echo ""
echo "🔧 Fix Option 3: Reinstall all requirements"
read -p "   Apply this fix? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt --upgrade
    echo "✅ Requirements reinstalled"
fi

# Test the fixes
echo ""
echo "🧪 Testing fixes..."
python test_simple.py 2>&1 | grep -E "(✅|❌|COMPLETED)"

echo ""
echo "=============================================="
echo "✅ Dependency fix complete!"
echo ""
echo "Next steps:"
echo "  1. Run: python test_simple.py"
echo "  2. Run: streamlit run ui/streamlit_app.py"
echo "  3. Check TESTING_GUIDE.md for more tests"