# CP2 Construction Site Selection Pipeline
# Based on methods guide for reproducible data processing

.PHONY: all base scenarios recs clean help

# Default target
all: base scenarios recs

# Base data processing pipeline
base: preprocess extrapolate build_fact compute_scores

# Preprocessing steps
preprocess:
	@echo "🔄 Running preprocessing steps..."
	python scripts/preprocess_grdp.py
	python scripts/preprocess_income.py
	python scripts/preprocess_housing_prices.py
	python scripts/preprocess_demand.py
	python scripts/preprocess_hazard.py
	python scripts/preprocess_labor.py
	@echo "✅ Preprocessing completed"

# Extrapolation steps
extrapolate:
	@echo "🔄 Running extrapolation steps..."
	python scripts/extrapolate_grdp.py
	python scripts/extrapolate_income.py
	python scripts/extrapolate_demand.py
	python scripts/extrapolate_labor.py
	@echo "✅ Extrapolation completed"

# Build unified fact table
build_fact:
	@echo "🔄 Building unified fact table..."
	python scripts/build_fact_table_2024.py
	@echo "✅ Fact table built"

# Compute base scores
compute_scores:
	@echo "🔄 Computing base scores..."
	python scripts/compute_scores_base.py
	@echo "✅ Base scores computed"

# Generate scenarios
scenarios:
	@echo "🔄 Generating scenarios..."
	python scripts/generate_app_table_with_scenarios.py \
		--in data/curated/with\ scores/fact_table_FULL_FINAL.csv \
		--out data/curated/with\ scores/app-ready/fact_table_app_READY_WITH_SCENARIOS.csv
	@echo "✅ Scenarios generated"

# Add recommendations
recs:
	@echo "🔄 Adding recommendations..."
	python scripts/add_recommendations.py
	@echo "✅ Recommendations added"

# Clean intermediate files
clean:
	@echo "🧹 Cleaning intermediate files..."
	rm -rf data/cleaned/*/
	rm -rf data/extrapolated/*/
	rm -f data/curated/fact_table_2024.csv
	rm -f data/curated/with\ scores/fact_table_FULL_FINAL.csv
	@echo "✅ Clean completed"

# Clean all generated files
clean-all: clean
	@echo "🧹 Cleaning all generated files..."
	rm -rf data/curated/with\ scores/app-ready/
	@echo "✅ Full clean completed"

# Help
help:
	@echo "CP2 Construction Site Selection Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Run complete pipeline (base + scenarios + recs)"
	@echo "  base       - Run base data processing (preprocess + extrapolate + build_fact + compute_scores)"
	@echo "  preprocess - Run all preprocessing scripts"
	@echo "  extrapolate - Run all extrapolation scripts"
	@echo "  build_fact - Build unified fact table"
	@echo "  compute_scores - Compute base scores with IPR"
	@echo "  scenarios  - Generate scenario variations"
	@echo "  recs       - Add client recommendations"
	@echo "  clean      - Remove intermediate files"
	@echo "  clean-all  - Remove all generated files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all           # Run complete pipeline"
	@echo "  make base          # Run base processing only"
	@echo "  make scenarios     # Generate scenarios from existing fact table"
	@echo "  make clean         # Clean intermediate files"
