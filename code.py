#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os
import math
warnings.filterwarnings('ignore')

# ============================================
# Part 1: Data Loading and Preprocessing
# ============================================

def load_transaction_data(file_path):
    """
    Load transaction data and convert to list format
    Each row in the CSV file is a transaction, with items separated by commas
    """
    try:
        # Attempt to read CSV directly
        df = pd.read_csv(file_path)

        # Convert each row to a list of items
        transactions = []
        for idx, row in df.iterrows():
            # Filter out null values, convert to string list
            items = [str(item).strip() for item in row.dropna() if str(item).strip() != '']
            if items:  # Only add non-empty transactions
                transactions.append(items)

        return transactions
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def get_dataset_statistics(transactions):
    """
    Calculate dataset statistics
    """
    num_transactions = len(transactions)
    all_items = set()
    transaction_lengths = []

    for trans in transactions:
        all_items.update(trans)
        transaction_lengths.append(len(trans))

    num_unique_items = len(all_items)
    avg_transaction_length = np.mean(transaction_lengths)

    return {
        'num_transactions': num_transactions,
        'num_unique_items': num_unique_items,
        'avg_transaction_length': avg_transaction_length
    }

# ============================================
# Part 2: Apriori Algorithm Time Measurement
# ============================================

def run_apriori_with_timing(transactions, min_support=0.01, min_confidence=0.5):
    """
    Run Apriori algorithm and measure time
    Returns: Frequent itemset generation time, rule generation time, total time
    """
    # Data transformation
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Measure frequent itemset generation time
    start_itemset = time.time()
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    end_itemset = time.time()
    itemset_time = end_itemset - start_itemset

    # Measure rule generation time
    start_rules = time.time()
    if len(frequent_itemsets) > 0:
        try:
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                     min_threshold=min_confidence)
        except:
            rules = pd.DataFrame()
    else:
        rules = pd.DataFrame()
    end_rules = time.time()
    rules_time = end_rules - start_rules

    total_time = itemset_time + rules_time

    return itemset_time, rules_time, total_time, len(frequent_itemsets), len(rules)

# ============================================
# Part 3: Improved Brute-Force Time Estimation
# ============================================

def estimate_bruteforce_time(num_unique_items, num_transactions):
    num_itemsets = 2**num_unique_items - 1
    time_per_check = 1e-6  
    estimated_time = num_itemsets * num_transactions * time_per_check

    return estimated_time

# ============================================
# Part 4: Main Program - Running Experiments
# ============================================

def main():
    # Dataset file paths
    dataset_files = [
        "~/Desktop/R/2025/1029/归档/dataset_1000trans.csv",
        "~/Desktop/R/2025/1029/归档/dataset_2000trans.csv", 
        "~/Desktop/R/2025/1029/归档/dataset_3000trans.csv",
        "~/Desktop/R/2025/1029/归档/dataset_4000trans.csv",
        "~/Desktop/R/2025/1029/归档/dataset_5000trans.csv"
    ]

    # Expand paths
    dataset_files = [os.path.expanduser(f) for f in dataset_files]

    # Store results
    results = {
        'dataset_name': [],
        'num_transactions': [],
        'num_unique_items': [],
        'avg_trans_length': [],
        'apriori_itemset_time': [],
        'apriori_rules_time': [], 
        'apriori_total_time': [],
        'num_frequent_itemsets': [],
        'num_rules': [],
        'bruteforce_estimated_time': []
    }

    # Apriori parameter settings
    min_support = 0.01
    min_confidence = 0.5

    # Reference point variables
    reference_items = None
    reference_transactions = None 
    reference_avg_length = None
    reference_apriori_time = None

    # Process each dataset
    for i, file_path in enumerate(dataset_files, 1):
        print(f"\nProcessing dataset {i}/{len(dataset_files)}: {os.path.basename(file_path)}")
        print("-" * 50)

        # Load data
        transactions = load_transaction_data(file_path)

        if transactions is None:
            print(f"  Error: Unable to load dataset {file_path}")
            continue

        # Get statistics
        stats = get_dataset_statistics(transactions)
        print(f"  Number of transactions: {stats['num_transactions']}")
        print(f"  Number of unique items: {stats['num_unique_items']}")
        print(f"  Average transaction length: {stats['avg_transaction_length']:.0f}")

        # Run Apriori algorithm
        print("  Running Apriori algorithm...")
        itemset_time, rules_time, total_time, num_itemsets, num_rules = \
            run_apriori_with_timing(transactions, min_support, min_confidence)

        print(f"  Apriori total time: {total_time:.4f} seconds")
        print(f"  Found frequent itemsets: {num_itemsets}, Generated rules: {num_rules}")

        # Set reference point (use first valid dataset)
        if reference_items is None:
            reference_items = stats['num_unique_items']
            reference_transactions = stats['num_transactions']
            reference_avg_length = stats['avg_transaction_length']
            reference_apriori_time = total_time
            print(f"  >> Set this dataset as reference point")

        # Estimate brute-force time
        bruteforce_time = estimate_bruteforce_time(
             stats['num_unique_items'],
             stats['num_transactions']
        )

        print(f"  Brute-force estimated time: {bruteforce_time:.2e} seconds")

        # Format time display
        if bruteforce_time >= 31536000:  # 1 year
            print(f"    (Approximately {bruteforce_time/31536000:.1f} years)")
        elif bruteforce_time >= 86400:    # 1 day
            print(f"    (Approximately {bruteforce_time/86400:.1f} days)")
        elif bruteforce_time >= 3600:     # 1 hour
            print(f"    (Approximately {bruteforce_time/3600:.1f} hours)")
        elif bruteforce_time >= 60:       # 1 minute
            print(f"    (Approximately {bruteforce_time/60:.1f} minutes)")

        speedup = bruteforce_time / total_time if total_time > 0 else 0
        print(f"  Apriori speedup: {speedup:.2e} times")

        # Save results
        results['dataset_name'].append(os.path.basename(file_path))
        results['num_transactions'].append(stats['num_transactions'])
        results['num_unique_items'].append(stats['num_unique_items'])
        results['avg_trans_length'].append(stats['avg_transaction_length'])
        results['apriori_itemset_time'].append(itemset_time)
        results['apriori_rules_time'].append(rules_time)
        results['apriori_total_time'].append(total_time)
        results['num_frequent_itemsets'].append(num_itemsets)
        results['num_rules'].append(num_rules)
        results['bruteforce_estimated_time'].append(bruteforce_time)

    # ============================================
    # Part 5: Create Visualization Charts
    # ============================================

    # Create professional charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Algorithm time comparison (log scale)
    datasets = [f"{t} trans" for t in results['num_transactions']]
    x_pos = np.arange(len(datasets))

    ax1.bar(x_pos - 0.2, results['apriori_total_time'], 0.4, 
            label='Apriori Algorithm', color='#3498db', alpha=0.8)
    ax1.bar(x_pos + 0.2, [min(t, 1e6) for t in results['bruteforce_estimated_time']], 0.4,
            label='Brute-Force (Estimated)', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Time Comparison\n(Apriori vs Brute-Force)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Chart 2: Apriori algorithm time breakdown
    ax2.bar(x_pos - 0.2, results['apriori_itemset_time'], 0.4,
            label='Frequent Itemset Generation', color='#2ecc71', alpha=0.8)
    ax2.bar(x_pos + 0.2, results['apriori_rules_time'], 0.4,
            label='Rule Generation', color='#f39c12', alpha=0.8)
    ax2.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Apriori Algorithm Time Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Chart 3: Speedup factors
    speedup_factors = [bf/ap for bf, ap in zip(results['bruteforce_estimated_time'], 
                                             results['apriori_total_time'])]
    bars = ax3.bar(x_pos, speedup_factors, color='#9b59b6', alpha=0.8)
    ax3.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Speedup Factor (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Apriori Speedup vs Brute-Force', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(datasets, rotation=45)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1e}', ha='center', va='bottom', fontweight='bold')

    # Chart 4: Mining results statistics
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(x_pos, results['num_frequent_itemsets'], 'o-', 
                    linewidth=2, markersize=8, label='Frequent Itemsets', color='#e67e22')
    line2 = ax4_twin.plot(x_pos, results['num_rules'], 's-', 
                         linewidth=2, markersize=8, label='Association Rules', color='#16a085')
    ax4.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Frequent Itemsets', fontsize=12, fontweight='bold', color='#e67e22')
    ax4_twin.set_ylabel('Number of Rules', fontsize=12, fontweight='bold', color='#16a085')
    ax4.set_title('Mining Results Statistics', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(datasets, rotation=45)
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig('association_rules_analysis.png', dpi=300, bbox_inches='tight')

    # ============================================
    # Part 6: Generate Results Table
    # ============================================

    # Create detailed results dataframe
    results_df = pd.DataFrame({
        'Dataset': results['dataset_name'],
        'Transactions': results['num_transactions'],
        'Unique_Items': results['num_unique_items'],
        'Avg_Length': [f"{x:.2f}" for x in results['avg_trans_length']],
        'Apriori_Time(s)': [f"{x:.4f}" for x in results['apriori_total_time']],
        'BruteForce_Est_Time(s)': [f"{x:.2e}" for x in results['bruteforce_estimated_time']],
        'Speedup_Factor': [f"{bf/ap:.2e}" for bf, ap in 
                         zip(results['bruteforce_estimated_time'], results['apriori_total_time'])],
        'Frequent_Itemsets': results['num_frequent_itemsets'],
        'Rules_Generated': results['num_rules']
    })

    print("\n" + "="*80)
    print("Experimental Summary Results")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save detailed results
    results_df.to_csv('association_rules_experiment_results.csv', index=False)

    return results_df

if __name__ == "__main__":
    results_df = main()


# In[ ]:




