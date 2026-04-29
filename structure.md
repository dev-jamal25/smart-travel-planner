# Project Structure

smart-travel-planner
├── .claude
│   └── settings.local.json
├── .github
│   └── workflows
│       ├── copilot-instructions.md
│       └── test.yml
├── backend
│   ├── app
│   │   ├── db
│   │   │   ├── migrations
│   │   │   │   ├── versions
│   │   │   │   │   └── c9fb57636433_create_runs_table.py
│   │   │   │   ├── env.py
│   │   │   │   ├── README
│   │   │   │   └── script.py.mako
│   │   │   ├── __init__.py
│   │   │   └── session.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   └── db.py
│   │   ├── routers
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── chat.py
│   │   │   └── history.py
│   │   ├── schemas
│   │   │   ├── __init__.py
│   │   │   └── auth.py
│   │   ├── services
│   │   │   └── __init__.py
│   │   ├── tools
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── logging_setup.py
│   │   └── main.py
│   ├── ml
│   │   ├── data
│   │   │   └── .gitkeep
│   │   └── train_classifier.py
│   ├── models
│   │   └── .gitkeep
│   ├── notebooks
│   │   └── .gitkeep
│   ├── tests
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_auth.py
│   │   └── test_health.py
│   ├── .python-version
│   ├── alembic.ini
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── uv.lock
├── docs
├── frontend
│   └── .gitkeep
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .structureignore
├── CLAUDE.md
├── docker-compose.yml
└── README.md
