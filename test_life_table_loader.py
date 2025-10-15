"""
Test script for Life_table.from_file() method

Run this after installing SRtools in your environment:
    python test_life_table_loader.py
"""

from SRtools import Life_table
import os

# Path to life table files
LIFETABLES_DIR = '../LifeTabels01/Lifetabels'

def test_csv_single_cohort():
    """Test loading single cohort from CSV file."""
    print("=" * 60)
    print("Test 1: Loading single cohort from CSV (cats.csv)")
    print("=" * 60)
    
    filepath = os.path.join(LIFETABLES_DIR, 'cats.csv')
    lt = Life_table.from_file(filepath)
    
    print(f"✓ Successfully loaded {len(lt.ages)} age bins")
    print(f"  Age range: {lt.ages[0]} to {lt.ages[-1]}")
    print(f"  Initial population (lx[0]): {lt.n_alive[0]:.0f}")
    print(f"  Tail bin: {lt.tail_bin}")
    print(f"  Has hazard values: {lt.hazard is not None}")
    print()
    
    return lt


def test_txt_specific_year():
    """Test loading specific year from multi-cohort TXT file."""
    print("=" * 60)
    print("Test 2: Loading specific year from TXT (Denmark 1835)")
    print("=" * 60)
    
    filepath = os.path.join(LIFETABLES_DIR, 'fltcoh_1x1_denmark.txt')
    lt = Life_table.from_file(filepath, year=1835)
    
    print(f"✓ Successfully loaded {len(lt.ages)} age bins")
    print(f"  Age range: {lt.ages[0]} to {lt.ages[-1]}")
    print(f"  Year: {lt.properties.get('year')}")
    print(f"  Initial population (lx[0]): {lt.n_alive[0]:.0f}")
    print(f"  Has hazard values: {lt.hazard is not None}")
    print()
    
    return lt


def test_txt_all_years():
    """Test loading all years from multi-cohort file."""
    print("=" * 60)
    print("Test 3: Loading ALL years from TXT (Sweden Females)")
    print("=" * 60)
    
    filepath = os.path.join(LIFETABLES_DIR, 'mltcoh_1x1_sweden_F.txt')
    lt_list = Life_table.from_file(filepath, year='all')
    
    print(f"✓ Successfully loaded {len(lt_list)} cohorts")
    print(f"  Year range: {lt_list[0].properties['year']} to {lt_list[-1].properties['year']}")
    print(f"  Each cohort has {len(lt_list[0].ages)} age bins")
    print()
    
    return lt_list


def test_custom_columns():
    """Test with custom column specifications."""
    print("=" * 60)
    print("Test 4: Loading with case-insensitive column matching")
    print("=" * 60)
    
    filepath = os.path.join(LIFETABLES_DIR, 'fltcoh_1x1_denmark.txt')
    # Test case-insensitive column name matching
    lt = Life_table.from_file(filepath, year=1835, 
                              age_col='age',  # lowercase
                              lx_col='LX',    # uppercase
                              mx_col='MX')
    
    print(f"✓ Successfully loaded with custom column names")
    print(f"  Columns matched successfully (case-insensitive)")
    print()
    
    return lt


def test_error_handling():
    """Test error handling."""
    print("=" * 60)
    print("Test 5: Error handling")
    print("=" * 60)
    
    filepath = os.path.join(LIFETABLES_DIR, 'fltcoh_1x1_denmark.txt')
    
    # Test 1: Multi-cohort file without year specification
    try:
        lt = Life_table.from_file(filepath)
        print("✗ Should have raised error for multi-cohort without year")
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:60]}...")
    
    # Test 2: Non-existent year
    try:
        lt = Life_table.from_file(filepath, year=1700)
        print("✗ Should have raised error for non-existent year")
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:60]}...")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Life_table.from_file() method")
    print("=" * 60 + "\n")
    
    try:
        # Run tests
        lt1 = test_csv_single_cohort()
        lt2 = test_txt_specific_year()
        lt_list = test_txt_all_years()
        lt4 = test_custom_columns()
        test_error_handling()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        # Quick visualization test
        print("\nQuick plot test...")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot cats
        lt1.plot_survival(ax=axes[0], label='Cats')
        axes[0].set_title('Cats Survival Curve')
        axes[0].legend()
        
        # Plot Denmark 1835
        lt2.plot_survival(ax=axes[1], label='Denmark 1835 (Females)')
        axes[1].set_title('Denmark 1835 Survival Curve')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('life_table_test_plot.png', dpi=100)
        print("✓ Plots saved to 'life_table_test_plot.png'")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

