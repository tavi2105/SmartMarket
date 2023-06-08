import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import csv

import warnings

warnings.filterwarnings('ignore')

# Loop the data lines
with open("./data/groceries.csv", 'r') as temp_f:
    # get No of columns in each line
    col_count = [len(l.split(",")) for l in temp_f.readlines()]

# Free memory space
del temp_f

# Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]

# Read csv
df = pd.read_csv("./data/groceries.csv", header=None, delimiter=",", names=column_names)

transactions = df.values.astype(str).tolist()
transactions = [[item for item in row if item != 'nan'] for row in transactions]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2.40)

rules.sort_values(by=['confidence'], ascending=False)


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
rules['rule'] = rules.index
rules.drop(['antecedent support', 'consequent support', 'conviction', 'antecedent_len', 'consequent_len'],
           axis=1).sort_values(by=['confidence'], ascending=False)

new_rules = rules.drop(['antecedent support',
                        'consequent support',
                        'conviction',
                        'confidence',
                        'support',
                        'leverage',
                        'zhangs_metric',
                        'rule',
                        'antecedent_len',
                        'consequent_len'], axis=1)[rules['antecedent_len'] == 1][rules['consequent_len'] == 1]


with open('./data/rules.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    for rule in new_rules.values:
        antecedent = [a for a in rule[0]][0]
        consequent = [a for a in rule[1]][0]
        writer.writerow([antecedent, consequent, rule[2]])


del f
