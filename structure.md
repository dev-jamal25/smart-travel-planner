# Project Structure

smart-travel-planner
├── .claude
│   └── settings.local.json
├── .github
│   ├── workflows
│   │   └── test.yml
│   └── copilot-instructions.md
├── backend
│   ├── app
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   ├── graph.py
│   │   │   ├── model_router.py
│   │   │   └── state.py
│   │   ├── db
│   │   │   ├── migrations
│   │   │   │   ├── versions
│   │   │   │   │   ├── a4f8c2d1e9b3_add_rag_tables.py
│   │   │   │   │   ├── c9fb57636433_create_runs_table.py
│   │   │   │   │   └── d7e2f4a9c1b6_add_agent_persistence_tables.py
│   │   │   │   ├── env.py
│   │   │   │   ├── README
│   │   │   │   └── script.py.mako
│   │   │   ├── repositories
│   │   │   │   ├── __init__.py
│   │   │   │   └── agent_runs.py
│   │   │   ├── __init__.py
│   │   │   └── session.py
│   │   ├── evaluation
│   │   │   ├── __init__.py
│   │   │   └── rag_eval_cases.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   └── db.py
│   │   ├── prompts
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── safety.py
│   │   │   └── synthesis.py
│   │   ├── routers
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── chat.py
│   │   │   ├── classifier.py
│   │   │   ├── history.py
│   │   │   ├── rag.py
│   │   │   ├── traces.py
│   │   │   ├── weather.py
│   │   │   └── webhook.py
│   │   ├── schemas
│   │   │   ├── __init__.py
│   │   │   ├── agent_tools.py
│   │   │   ├── agent.py
│   │   │   ├── auth.py
│   │   │   ├── classifier.py
│   │   │   ├── rag.py
│   │   │   ├── traces.py
│   │   │   ├── weather.py
│   │   │   └── webhook.py
│   │   ├── services
│   │   │   ├── __init__.py
│   │   │   ├── agent_service.py
│   │   │   ├── classifier_service.py
│   │   │   ├── destination_coordinates.py
│   │   │   ├── destination_profiles.py
│   │   │   ├── rag_service.py
│   │   │   ├── weather_service.py
│   │   │   └── webhook_service.py
│   │   ├── tools
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── classifier_tool.py
│   │   │   ├── rag_tool.py
│   │   │   ├── registry.py
│   │   │   └── weather_tool.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── lifespan.py
│   │   ├── logging_setup.py
│   │   ├── main.py
│   │   ├── retries.py
│   │   └── tracing.py
│   ├── ml
│   │   ├── data
│   │   │   ├── processed
│   │   │   │   └── travel_data_labeled.csv
│   │   │   └── raw
│   │   │       └── Tourist_Destinations.csv
│   │   ├── outputs
│   │   │   ├── final_classification_report.json
│   │   │   ├── final_confusion_matrix.json
│   │   │   ├── final_test_metrics.json
│   │   │   ├── logistic_regression_top_coefficients.json
│   │   │   ├── model_metadata.json
│   │   │   ├── permutation_importance_best_validation_model.json
│   │   │   ├── results_cv.json
│   │   │   ├── results_validation.json
│   │   │   ├── results.csv
│   │   │   ├── results.json
│   │   │   └── tuning_results.json
│   │   ├── README.md
│   │   └── train_classifier.py
│   ├── models
│   │   └── final_travel_style_pipeline.joblib
│   ├── notebooks
│   │   ├── travel_cleaned_labeled.ipynb
│   │   └── Travel_transform.ipynb
│   ├── rag_data
│   │   ├── eval
│   │   │   └── retrieval_eval_cases.json
│   │   └── raw
│   │       ├── bali_details.txt
│   │       ├── bali_overview.txt
│   │       ├── banff_details.txt
│   │       ├── banff_overview.txt
│   │       ├── dubai_details.txt
│   │       ├── dubai_overview.txt
│   │       ├── interlaken_details.txt
│   │       ├── interlaken_overview.txt
│   │       ├── istanbul_details.txt
│   │       ├── istanbul_overview.txt
│   │       ├── kraków_details.txt
│   │       ├── kraków_overview.txt
│   │       ├── kyoto_details.txt
│   │       ├── kyoto_overview.txt
│   │       ├── santorini_details.txt
│   │       ├── santorini_overview.txt
│   │       ├── singapore_details.txt
│   │       ├── singapore_overview.txt
│   │       ├── tbilisi_details.txt
│   │       └── tbilisi_overview.txt
│   ├── scripts
│   │   ├── fetch_wikivoyage_raw.py
│   │   └── ingest_rag_documents.py
│   ├── tests
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_agent_graph.py
│   │   ├── test_agent_persistence.py
│   │   ├── test_agent_schemas.py
│   │   ├── test_agent_tools.py
│   │   ├── test_auth.py
│   │   ├── test_chat.py
│   │   ├── test_classifier.py
│   │   ├── test_destination_profiles.py
│   │   ├── test_health.py
│   │   ├── test_langsmith_tracing.py
│   │   ├── test_model_router.py
│   │   ├── test_paths.py
│   │   ├── test_rag_endpoints.py
│   │   ├── test_rag_service.py
│   │   ├── test_traces.py
│   │   ├── test_weather.py
│   │   └── test_webhook.py
│   ├── .dockerignore
│   ├── .env.example
│   ├── .python-version
│   ├── alembic.ini
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── pyproject.toml
│   └── uv.lock
├── docs
│   ├── screenshots
│   │   └── 04-langsmith-trace.png
│   ├── agent_orchestration.md
│   ├── ai_change_log.txt
│   ├── discord_webhook.md
│   ├── rag_foundation.md
│   ├── rag_retrieval_tests.md
│   └── weather_tool.md
├── frontend
│   ├── .vite
│   │   └── deps
│   │       ├── _metadata.json
│   │       ├── @supabase_supabase-js.js
│   │       ├── @supabase_supabase-js.js.map
│   │       ├── chunk-BUSYA2B4.js
│   │       ├── chunk-BUSYA2B4.js.map
│   │       ├── chunk-JCH2SJW3.js
│   │       ├── chunk-JCH2SJW3.js.map
│   │       ├── package.json
│   │       ├── react_jsx-dev-runtime.js
│   │       ├── react_jsx-dev-runtime.js.map
│   │       ├── react-dom_client.js
│   │       ├── react-dom_client.js.map
│   │       ├── react.js
│   │       └── react.js.map
│   ├── src
│   │   ├── components
│   │   │   ├── AuthForm.tsx
│   │   │   ├── ChatPlanner.tsx
│   │   │   ├── RunHistory.tsx
│   │   │   └── TracePanel.tsx
│   │   ├── lib
│   │   │   ├── api.ts
│   │   │   └── supabase.ts
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   └── vite-env.d.ts
│   ├── .dockerignore
│   ├── .env
│   ├── .env.example
│   ├── Dockerfile
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .structureignore
├── CLAUDE.md
├── docker-compose.yml
└── README.md
