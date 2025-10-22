#!/usr/bin/env python3
"""
Identify and optionally remove scripts that are not related to current output files.
"""

import os
import shutil
from pathlib import Path


def main():
    # Scripts that ARE related to current outputs (keep these)
    keep_scripts = {
        'generate_app_table_with_scenarios.py',  # Matches fact_table_app_READY_WITH_SCENARIOS.csv
        'lfs_forecast_ets_only.py',             # Matches lfs_city_monthly_2024_PROJECTED_ETS.csv
        'lfs_forecast_ts_bakeoff.py',           # Matches lfs_city_monthly_2024_PROJECTED_TS.csv
        'lfs_city_builder.py',                  # Matches lfs_city_monthly_agg_2024.csv
        'utils.py',                             # New utility module
        'preprocess_grdp.py',                   # New preprocessing script
        'preprocess_income.py',                 # New preprocessing script
        'preprocess_housing_prices.py',         # New preprocessing script
        'preprocess_demand.py',                 # New preprocessing script
        'preprocess_hazard.py',                 # New preprocessing script
        'build_fact_table_2024.py',             # New integration script
        'compute_scores_base.py',               # New scoring script
        'add_recommendations.py',               # New recommendations script
        'cleanup_unused_scripts.py',            # This script
    }
    
    # Scripts that are NOT related to current outputs (can be removed)
    unused_scripts = {
        'forecast_city_series.py',              # Generic forecasting tool
        'project_population_from_drivers.py',   # Population projection, no matching output
        'project_units_from_drivers.py',        # Housing units projection, no matching output
        'project_prices_from_drivers.py',       # Price projection, no matching output
        'backtest.py',                          # Utility for forecasting
        'lfs_trend_visualization.py',           # Visualization script, no matching output
        'panel_prep.py',                        # Utility script
        'utils_guardrails.py',                  # Utility functions (replaced by utils.py)
        'env_check.py',                         # Environment check utility
    }
    
    scripts_dir = Path('scripts')
    
    print("🔍 Analyzing scripts in scripts/ directory...")
    
    # Find all Python files in scripts directory
    all_scripts = set(f.name for f in scripts_dir.glob('*.py'))
    
    print(f"\n📊 Script Analysis:")
    print(f"   - Total Python scripts: {len(all_scripts)}")
    print(f"   - Scripts to keep: {len(keep_scripts)}")
    print(f"   - Scripts that can be removed: {len(unused_scripts)}")
    
    # Find scripts that are not in either category
    unknown_scripts = all_scripts - keep_scripts - unused_scripts
    if unknown_scripts:
        print(f"   - Unknown scripts (not categorized): {len(unknown_scripts)}")
        print(f"     {list(unknown_scripts)}")
    
    print(f"\n✅ Scripts to KEEP (related to outputs):")
    for script in sorted(keep_scripts):
        if script in all_scripts:
            print(f"   ✓ {script}")
        else:
            print(f"   ? {script} (not found)")
    
    print(f"\n❌ Scripts that can be REMOVED (not related to outputs):")
    for script in sorted(unused_scripts):
        if script in all_scripts:
            print(f"   ✗ {script}")
        else:
            print(f"   - {script} (not found)")
    
    # Ask user if they want to remove unused scripts
    print(f"\n🗑️  Removal Options:")
    print("   1. Move unused scripts to a backup folder")
    print("   2. Delete unused scripts permanently")
    print("   3. Just show the analysis (no action)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        # Move to backup folder
        backup_dir = Path('scripts/backup_unused')
        backup_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        for script in unused_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                backup_path = backup_dir / script
                shutil.move(str(script_path), str(backup_path))
                print(f"   📦 Moved {script} to backup/")
                moved_count += 1
        
        print(f"\n✅ Moved {moved_count} unused scripts to scripts/backup_unused/")
        
    elif choice == '2':
        # Delete permanently
        deleted_count = 0
        for script in unused_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                script_path.unlink()
                print(f"   🗑️  Deleted {script}")
                deleted_count += 1
        
        print(f"\n✅ Deleted {deleted_count} unused scripts permanently")
        
    else:
        print("\n📋 Analysis complete. No files were modified.")
    
    print(f"\n📁 Current scripts directory contents:")
    remaining_scripts = sorted([f.name for f in scripts_dir.glob('*.py')])
    for script in remaining_scripts:
        print(f"   - {script}")


if __name__ == "__main__":
    main()
