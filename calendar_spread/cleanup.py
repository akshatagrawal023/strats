"""
Cleanup script to remove unnecessary files from calendar_spread folder
"""

import os
import shutil

def cleanup_calendar_spread():
    """Remove unnecessary files from calendar_spread folder"""
    
    # Files to keep (essential)
    keep_files = {
        '__init__.py',
        'nifty50_processor.py',
        'historical_data_fetcher.py', 
        'data_viewer.py',
        'historical_data.json'
    }
    
    # Files to remove
    remove_files = {
        'backtest.py',
        'calendar_spread_strategy.py',
        'calendar_trading_system.py',
        'config.py',
        'data_processor.py',
        'hist_data.py',
        'historical_data_manager.py',
        'lot_size_processor.py',
        'position_monitor.py',
        'README.md',
        'results_analyzer.py',
        'run_backtest.py',
        'setup_nifty50_backtest.py',
        'SIMPLE_GUIDE.md',
        'test_processor.py',
        'trade_manager.py',
        'trading_state.py',
        'underlying_scanner.py'
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🧹 CLEANING UP CALENDAR_SPREAD FOLDER")
    print("=" * 50)
    
    # List current files
    current_files = set(os.listdir(current_dir))
    print(f"Current files: {len(current_files)}")
    
    # Remove unnecessary files
    removed_count = 0
    for file in remove_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {file}: {e}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # List remaining files
    remaining_files = set(os.listdir(current_dir))
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   Files removed: {removed_count}")
    print(f"   Files remaining: {len(remaining_files)}")
    
    print(f"\n📁 REMAINING FILES:")
    for file in sorted(remaining_files):
        if file in keep_files:
            print(f"   ✅ {file}")
        else:
            print(f"   ⚠️  {file} (not in keep list)")
    
    print("\n✅ Cleanup completed!")

if __name__ == "__main__":
    cleanup_calendar_spread()
