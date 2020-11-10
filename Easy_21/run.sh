rm -r fig
rm -r log
mkdir fig
mkdir log
python3 value_iter.py
python3 policy_iter.py
python3 SARSA.py
python3 Q_learning.py
python3 monte_carlo.py
python3 monte_carlo_optimistic.py
