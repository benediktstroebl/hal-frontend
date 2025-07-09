import json
from pathlib import Path
import sqlite3
import pickle
from functools import lru_cache
import threading
import pandas as pd
import ast
from scipy import stats
import yaml
import numpy as np

# Define column schemas
PARSED_RESULTS_COLUMNS = {
    'benchmark_name': 'TEXT',
    'agent_name': 'TEXT', 
    'date': 'TEXT',
    'run_id': 'TEXT',
    'successful_tasks': 'TEXT',
    'failed_tasks': 'TEXT',
    'total_cost': 'REAL',
    'accuracy': 'REAL',
    'precision': 'REAL',
    'recall': 'REAL',
    'f1_score': 'REAL',
    'auc': 'REAL',
    'overall_score': 'REAL',
    'vectorization_score': 'REAL',
    'fathomnet_score': 'REAL',
    'feedback_score': 'REAL',
    'house_price_score': 'REAL',
    'spaceship_titanic_score': 'REAL',
    'amp_parkinsons_disease_progression_prediction_score': 'REAL',
    'cifar10_score': 'REAL',
    'imdb_score': 'REAL',
    'level_1_accuracy': 'REAL',
    'level_2_accuracy': 'REAL',
    'level_3_accuracy': 'REAL',
    'task_goal_completion': 'REAL',  # New column
    'scenario_goal_completion': 'REAL',  # New column
    'combined_scorer_inspect_evals_avg_refusals': 'REAL',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'REAL',
    'accuracy_ci': 'TEXT',  # Using TEXT since it stores formatted strings like "-0.123/+0.456"
    'cost_ci': 'TEXT',
}

# Define which columns should be included in aggregation and how
AGGREGATION_RULES = {
    'date': 'first',
    'total_cost': 'mean',
    'accuracy': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'f1_score': 'mean',
    'auc': 'mean',
    'overall_score': 'mean',
    'vectorization_score': 'mean',
    'fathomnet_score': 'mean',
    'feedback_score': 'mean',
    'house_price_score': 'mean',
    'spaceship_titanic_score': 'mean',
    'amp_parkinsons_disease_progression_prediction_score': 'mean',
    'cifar10_score': 'mean',
    'imdb_score': 'mean',
    'level_1_accuracy': 'mean',
    'level_2_accuracy': 'mean',
    'level_3_accuracy': 'mean',
    'task_goal_completion': 'mean',
    'scenario_goal_completion': 'mean',
    'combined_scorer_inspect_evals_avg_refusals': 'mean',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'mean',
    'Verified': 'first',
    'Runs': 'first',
    'Traces': 'first',
    'accuracy_ci': 'first',
    'cost_ci': 'first',
    'run_id': 'first',
}

# Define column display names
COLUMN_DISPLAY_NAMES = {
    'agent_name': 'Agent Name',
    'url': 'URL',
    'date': 'Date',
    'total_cost': 'Total Cost',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1 Score',
    'auc': 'AUC',
    'overall_score': 'Overall Score',
    'vectorization_score': 'Vectorization Score',
    'fathomnet_score': 'Fathomnet Score',
    'feedback_score': 'Feedback Score',
    'house_price_score': 'House Price Score',
    'spaceship_titanic_score': 'Spaceship Titanic Score',
    'amp_parkinsons_disease_progression_prediction_score': 'AMP Parkinsons Disease Progression Prediction Score',
    'cifar10_score': 'CIFAR10 Score',
    'imdb_score': 'IMDB Score',
    'level_1_accuracy': 'Level 1 Accuracy',
    'level_2_accuracy': 'Level 2 Accuracy',
    'level_3_accuracy': 'Level 3 Accuracy',
    'task_goal_completion': 'Task Goal Completion',
    'scenario_goal_completion': 'Scenario Goal Completion',
    'accuracy_ci': 'Accuracy CI',
    'cost_ci': 'Total Cost CI',
    'combined_scorer_inspect_evals_avg_refusals': 'Refusals',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'Non-Refusal Harm Score',
}

DEFAULT_PRICING = {
    "text-embedding-3-small": {"prompt_tokens": 0.02, "completion_tokens": 0},
    "text-embedding-3-large": {"prompt_tokens": 0.13, "completion_tokens": 0},
    "gpt-4o-2024-05-13": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "gpt-4o-2024-08-06": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "gpt-4o-2024-11-20": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
    "gpt-3.5-turbo": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
    "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
    "gpt-4-turbo": {"prompt_tokens": 10, "completion_tokens": 30},
    "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
    "o1-2024-12-17": {"prompt_tokens": 15, "completion_tokens": 60},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5, "completion_tokens": 15},
    "meta-llama/Llama-3-70b-chat-hf": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "deepseek-ai/deepseek-coder-33b-instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18}, # these are set to the llama 8n prices
    "gpt-4o": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
    "o1-mini-2024-09-12": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
    "o3-mini-2025-01-14": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3-mini-2025-01-31": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
    "claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3, "completion_tokens": 15},
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3, "completion_tokens": 15},
    "claude-3-5-haiku-20241022": {"prompt_tokens": 0.8, "completion_tokens": 4},
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"prompt_tokens": 0.8, "completion_tokens": 4},
    "openai/gpt-4o-2024-11-20": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "openai/gpt-4o-2024-08-06": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "openai/gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "openai/gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
    "openai/o1-mini-2024-09-12": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "openai/o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
    "openai/o1-2024-12-17": {"prompt_tokens": 15, "completion_tokens": 60},
    "openai/o3-mini-2025-01-14": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "openai/o3-mini-2025-01-31": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "anthropic/claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
    "anthropic/claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
    "google/gemini-1.5-pro": {"prompt_tokens": 1.25, "completion_tokens": 5},
    "google/gemini-1.5-flash": {"prompt_tokens": 0.075, "completion_tokens": 0.3},
    "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5, "completion_tokens": 3.5},
    "Meta-Llama-3.3-70B-Instruct-Turbo": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "us.meta.llama3-3-70b-instruct-v1:0": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5, "completion_tokens": 3.5},
    "together/meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "claude-3-7-sonnet-20250219": {"prompt_tokens": 3, "completion_tokens": 15},
    "anthropic/claude-3-7-sonnet-20250219": {"prompt_tokens": 3, "completion_tokens": 15},
}

class TracePreprocessor:
    def __init__(self, db_dir='preprocessed_traces'):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        self.local = threading.local()
        self.connections = {}

    def get_conn(self, benchmark_name):
        # Sanitize benchmark name for filename
        safe_name = benchmark_name.replace('/', '_').replace('\\', '_')
        db_path = self.db_dir / f"{safe_name}.db"
        
        # Get thread-specific connections dictionary
        if not hasattr(self.local, 'connections'):
            self.local.connections = {}
            
        # Create new connection if not exists for this benchmark
        if safe_name not in self.local.connections:
            self.local.connections[safe_name] = sqlite3.connect(db_path)
            
        return self.local.connections[safe_name]

    def create_tables(self, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            # Create parsed_results table dynamically from schema
            columns = [f"{col} {dtype}" for col, dtype in PARSED_RESULTS_COLUMNS.items()]
            create_parsed_results = f'''
                CREATE TABLE IF NOT EXISTS parsed_results (
                    {', '.join(columns)},
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            '''
            conn.execute(create_parsed_results)
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preprocessed_traces (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    date TEXT,
                    run_id TEXT,
                    raw_logging_results BLOB,
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS failure_reports (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    date TEXT,
                    run_id TEXT,
                    failure_report BLOB,
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    run_id TEXT,
                    model_name TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    input_tokens_cache_write INTEGER,
                    input_tokens_cache_read INTEGER,
                    PRIMARY KEY (benchmark_name, agent_name, run_id, model_name)
                )
            ''')

    def preprocess_traces(self, processed_dir="evals_live"):
        processed_dir = Path(processed_dir)
        for file in processed_dir.glob('*.json'):
            print(f"Processing {file}")
            with open(file, 'r') as f:
                data = json.load(f)
                agent_name = data['config']['agent_name']
                benchmark_name = data['config']['benchmark_name']
                if "inspect" in benchmark_name:
                    benchmark_name = benchmark_name.split("/")[-1]
                date = data['config']['date']
                config = data['config']

            # Create tables for this benchmark if they don't exist
            self.create_tables(benchmark_name)

            try:
                # raw_logging_results = pickle.dumps(data['raw_logging_results'])
                with self.get_conn(benchmark_name) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO preprocessed_traces 
                        (benchmark_name, agent_name, date, run_id) 
                        VALUES (?, ?, ?, ?)
                    ''', (benchmark_name, agent_name, date, config['run_id']))
            except Exception as e:
                print(f"Error preprocessing raw_logging_results in {file}: {e}")

            try:
                failure_report = pickle.dumps(data['failure_report'])
                with self.get_conn(benchmark_name) as conn:
                    conn.execute('''
                        INSERT INTO failure_reports 
                        (benchmark_name, agent_name, date, run_id, failure_report)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (benchmark_name, agent_name, date, config['run_id'], failure_report))
            except Exception as e:
                print(f"Error preprocessing failure_report in {file}: {e}")

            try:
                results = data['results']
                with self.get_conn(benchmark_name) as conn:
                    # Dynamically create placeholders and values list
                    columns = [col for col in PARSED_RESULTS_COLUMNS.keys() 
                              if col not in ['benchmark_name', 'agent_name', 'date', 'run_id']]
                    placeholders = ','.join(['?'] * (len(columns) + 4))  # +4 for benchmark_name, agent_name, date, run_id
                    
                    values = [
                        benchmark_name,
                        agent_name,
                        config['date'],
                        config['run_id']
                    ] + [str(results.get(col)) if col in ['successful_tasks', 'failed_tasks'] 
                         else results.get(col) for col in columns]

                    query = f'''
                        INSERT INTO parsed_results 
                        ({', '.join(PARSED_RESULTS_COLUMNS.keys())})
                        VALUES ({placeholders})
                    '''
                    conn.execute(query, values)
            except Exception as e:
                print(f"Error preprocessing parsed results in {file}: {e}")

            try:
                total_usage = data.get('total_usage', {})
                for model_name, usage in total_usage.items():
                    with self.get_conn(benchmark_name) as conn:
                        conn.execute('''
                            INSERT INTO token_usage 
                            (benchmark_name, agent_name, run_id, model_name, 
                            prompt_tokens, completion_tokens, input_tokens, output_tokens, total_tokens,
                            input_tokens_cache_write, input_tokens_cache_read)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            benchmark_name,
                            agent_name,
                            config['run_id'],
                            model_name,
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0),
                            usage.get('input_tokens', 0),
                            usage.get('output_tokens', 0),
                            usage.get('total_tokens', 0),
                            usage.get('input_tokens_cache_write', 0),
                            usage.get('input_tokens_cache_read', 0)
                        ))
            except Exception as e:
                print(f"Error preprocessing token usage in {file}: {e}")

    @lru_cache(maxsize=100)
    def get_analyzed_traces(self, agent_name, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, raw_logging_results, date FROM preprocessed_traces 
                WHERE benchmark_name = ? AND agent_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name, agent_name))

        # check for each row if raw_logging_results is not None
        df = df[df['raw_logging_results'].apply(lambda x: pickle.loads(x) is not None and x != 'None')]

        if len(df) == 0:
            return None

        # select latest run
        df = df.sort_values('date', ascending=False).groupby('agent_name').first().reset_index()

        return pickle.loads(df['raw_logging_results'][0])

    @lru_cache(maxsize=100)
    def get_failure_report(self, agent_name, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, date, failure_report FROM failure_reports 
                WHERE benchmark_name = ? AND agent_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name, agent_name))

        df = df[df['failure_report'].apply(lambda x: pickle.loads(x) is not None and x != 'None')]

        if len(df) == 0:
            return None

        df = df.sort_values('date', ascending=False).groupby('agent_name').first().reset_index()

        return pickle.loads(df['failure_report'][0])

    def _calculate_ci(self, data, confidence=0.95, type='minmax'):
        data = data[np.isfinite(data)]

        if len(data) < 2:
            return '', '', '' # No CI for less than 2 samples
        n = len(data)

        mean = np.mean(data)

        if type == 't':
            sem = stats.sem(data)
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)

        elif type == 'minmax':
            min = np.min(data)
            max = np.max(data)
            ci = (min, max)
        return mean, ci[0], ci[1]
    
    def get_parsed_results(self, benchmark_name, aggregate=True):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT * FROM parsed_results 
                WHERE benchmark_name = ?
                ORDER BY accuracy DESC
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))

        # Load metadata
        with open('agents_metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Create URL mapping
        url_mapping = {}
        if benchmark_name in metadata:
            for agent in metadata[benchmark_name]:
                if 'url' in agent and agent['url']:  # Only add if URL exists and is not empty
                    url_mapping[agent['agent_name']] = agent['url']

        # Add 'Verified' column
        verified_agents = self.load_verified_agents()
        df['Verified'] = df.apply(lambda row: '✓' if (benchmark_name, row['agent_name']) in verified_agents else '', axis=1)

        # Add URLs to agent names if they exist
        df['url'] = df['agent_name'].apply(lambda x: url_mapping.get(x, ''))

        # Add column for how many times an agent_name appears in the DataFrame
        df['Runs'] = df.groupby('agent_name')['agent_name'].transform('count')

        # Compute the 95% confidence interval for accuracy and cost for agents that have been run more than once
        df['accuracy_ci'] = None
        df['cost_ci'] = None

        # Before dropping run_id, create new column from it with download link
        # First create a temporary dataframe with agent_name and max accuracy
        max_accuracy_df = df.groupby('agent_name')['accuracy'].transform('max')
        # Create mask for rows with max accuracy in their group
        max_accuracy_mask = df['accuracy'] == max_accuracy_df
        # Create the Traces column, only setting values for max accuracy rows
        df['Traces'] = ''
        df.loc[max_accuracy_mask, 'Traces'] = df.loc[max_accuracy_mask, 'run_id'].apply(
            lambda x: f'https://huggingface.co/datasets/agent-evals/agent_traces/resolve/main/{x}.zip?download=true'
            if x else ''
        )
        
        df = df.drop(columns=['successful_tasks', 'failed_tasks'], axis=1)
        
        if aggregate:
            df = df.groupby('agent_name').agg(AGGREGATION_RULES).reset_index()
            
        # Rename columns using the display names mapping
        df = df.rename(columns=COLUMN_DISPLAY_NAMES)
        
        # Multiply accuracy by 100
        df['Accuracy'] = df['Accuracy'] * 100
        df['Scenario Goal Completion'] = df['Scenario Goal Completion'] * 100
        df['Task Goal Completion'] = df['Task Goal Completion'] * 100
        df['Level 1 Accuracy'] = df['Level 1 Accuracy'] * 100
        df['Level 2 Accuracy'] = df['Level 2 Accuracy'] * 100
        df['Level 3 Accuracy'] = df['Level 3 Accuracy'] * 100
        df['Refusals'] = df['Refusals'] * 100
        df['Non-Refusal Harm Score'] = df['Non-Refusal Harm Score'] * 100
        
        return df
    
    def get_task_success_data(self, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, accuracy, successful_tasks, failed_tasks
                FROM parsed_results 
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        
        # Get all unique task IDs
        task_ids = set()
        for tasks in df['successful_tasks']:
            if ast.literal_eval(tasks) is not None:
                task_ids.update(ast.literal_eval(tasks))
        for tasks in df['failed_tasks']:
            if ast.literal_eval(tasks) is not None:
                task_ids.update(ast.literal_eval(tasks))

        # Create a DataFrame with agent_name, task_ids, and success rates
        data_list = []
        for task_id in task_ids:
            for agent_name in df['agent_name'].unique():
                agent_runs = df[df['agent_name'] == agent_name]
                # Count how many times this task was successful across all runs
                successes = sum(1 for _, row in agent_runs.iterrows() 
                              if ast.literal_eval(row['successful_tasks']) is not None 
                              and task_id in ast.literal_eval(row['successful_tasks']))
                total_runs = len(agent_runs)
                success_rate = successes / total_runs if total_runs > 0 else 0
                
                data_list.append({
                    'agent_name': agent_name,
                    'task_id': task_id,
                    'success': success_rate
                })
        df = pd.DataFrame(data_list)

        df = df.rename(columns={
            'agent_name': 'Agent Name',
            'task_id': 'Task ID',
            'success': 'Success'
        })

        return df
    
    def load_verified_agents(self, file_path='agents_metadata.yaml'):
        with open(file_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        verified_agents = set()
        for benchmark, agents in metadata.items():
            for agent in agents:
                if 'verification_date' in agent:  # Only add if verified
                    verified_agents.add((benchmark, agent['agent_name']))
        
        return verified_agents

    def get_token_usage_with_costs(self, benchmark_name, pricing_config=None):
        """Get token usage data with configurable pricing"""
        if pricing_config is None:
            pricing_config = DEFAULT_PRICING

        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, model_name, run_id,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(input_tokens_cache_write) as input_tokens_cache_write,
                SUM(input_tokens_cache_read) as input_tokens_cache_read
                FROM token_usage
                WHERE benchmark_name = ?
                GROUP BY agent_name, model_name, run_id
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
                    
        # Calculate costs based on pricing config (prices are per 1M tokens)
        df['total_cost'] = 0.0
        for model, prices in pricing_config.items():
            mask = df['model_name'] == model
            df.loc[mask, 'total_cost'] = (
                df.loc[mask, 'input_tokens'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'output_tokens'] * prices['completion_tokens'] / 1e6 +
                df.loc[mask, 'input_tokens_cache_read'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'input_tokens_cache_write'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'prompt_tokens'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'completion_tokens'] * prices['completion_tokens'] / 1e6
            )
            
        # Sum total_cost for each run_id (if agents use multiple models, this will be the total cost for that run)
        df_temp = df.groupby('run_id')['total_cost'].sum().reset_index()
        df_temp = df_temp.rename(columns={'total_cost': 'total_cost_temp'})
        df = df.merge(df_temp, on='run_id', how='left')
        df['total_cost'] = df['total_cost_temp']
        df = df.drop('total_cost_temp', axis=1)
                                
        return df

    def get_parsed_results_with_costs(self, benchmark_name, pricing_config=None, aggregate=False):
        """Get parsed results with recalculated costs based on token usage"""
        # Get base results with URLs
        results_df = self.get_parsed_results(benchmark_name, aggregate=False)
        benchmark_name = results_df['benchmark_name'].iloc[0]
        
        # Get token usage with new costs
        token_costs = self.get_token_usage_with_costs(benchmark_name, pricing_config)
        
        for agent_name in results_df['Agent Name'].unique():
            agent_df = results_df[results_df['Agent Name'] == agent_name]
            
            if agent_name not in token_costs['agent_name'].unique():
                token_costs_df = results_df[results_df['Agent Name'] == agent_name]
            else:
                token_costs_df = token_costs[token_costs['agent_name'] == agent_name]
                
            if len(agent_df) > 1:
                accuracy_mean, accuracy_lower, accuracy_upper = self._calculate_ci(agent_df['Accuracy'], type='minmax')
                if agent_name not in token_costs['agent_name'].unique():
                    cost_mean, cost_lower, cost_upper = self._calculate_ci(token_costs_df['Total Cost'], type='minmax')
                else:
                    cost_mean, cost_lower, cost_upper = self._calculate_ci(token_costs_df['total_cost'], type='minmax')
                
                # Round CI values to 2 decimals
                accuracy_ci = f"-{abs(accuracy_mean - accuracy_lower):.2f}/+{abs(accuracy_mean - accuracy_upper):.2f}"
                cost_ci = f"-{abs(cost_mean - cost_lower):.2f}/+{abs(cost_mean - cost_upper):.2f}"
                
                results_df.loc[results_df['Agent Name'] == agent_name, 'Accuracy CI'] = accuracy_ci
                results_df.loc[results_df['Agent Name'] == agent_name, 'Total Cost CI'] = cost_ci
            
            if agent_name == 'Inspect ReAct Agent (o3-mini-2025-01-14)':
                results_df.loc[results_df['Agent Name'] == agent_name, 'Total Cost CI'] = "lower bound: $16.04*"
    
        # Group token costs by agent
        agent_costs = token_costs.groupby('agent_name')['total_cost'].mean().reset_index()

        agent_costs = agent_costs.rename(columns={
            'agent_name': 'agent_name_temp',
            'total_cost': 'Total Cost'
        })
                        
        # Drop existing Total Cost column if it exists
        if 'Total Cost' in results_df.columns:
            results_df['total_cost_temp'] = results_df['Total Cost']
            results_df = results_df.drop('Total Cost', axis=1)
            
        # Create temp column for matching, preserving the original Agent Name with URL
        results_df['agent_name_temp'] = results_df['Agent Name'].apply(lambda x: x.split('[')[1].split(']')[0] if '[' in x else x)
        
        # Update costs in results
        results_df = results_df.merge(agent_costs, on='agent_name_temp', how='left')
                
        # if Total Cost is NaN, set it to the value from total_cost_temp if it exists
        results_df['Total Cost'] = results_df['Total Cost'].fillna(results_df['total_cost_temp'])
        
        # If there is no token usage data, set Total Cost to total_cost from results key
        if len(token_costs) < 1:
            results_df['Total Cost'] = results_df['Total Cost'].fillna(results_df['total_cost_temp'])
                
        # Drop temp column
        results_df = results_df.drop('agent_name_temp', axis=1)
        results_df = results_df.drop('total_cost_temp', axis=1)
        
                    
        if aggregate:
            # Aggregate results while preserving URLs in Agent Name
            results_df = results_df.groupby('Agent Name', as_index=False).agg({
                'Date': 'first',
                'Total Cost': 'mean',
                'Accuracy': 'mean',
                'Precision': 'mean',
                'Recall': 'mean',
                'F1 Score': 'mean',
                'AUC': 'mean',
                'Overall Score': 'mean',
                'Vectorization Score': 'mean',
                'Fathomnet Score': 'mean',
                'Feedback Score': 'mean',
                'House Price Score': 'mean',
                'Spaceship Titanic Score': 'mean',
                'AMP Parkinsons Disease Progression Prediction Score': 'mean',
                'CIFAR10 Score': 'mean',
                'IMDB Score': 'mean',
                'Scenario Goal Completion': 'mean',
                'Task Goal Completion': 'mean',
                'Level 1 Accuracy': 'mean',
                'Level 2 Accuracy': 'mean',
                'Level 3 Accuracy': 'mean',
                'Verified': 'first',
                'Traces': 'first',
                'Runs': 'first',
                'Accuracy CI': 'first',
                'Total Cost CI': 'first',
                'URL': 'first',
                'Refusals': 'mean',
                'Non-Refusal Harm Score': 'mean'  # Preserve URL
            })
        
        # Round float columns to 2 decimal places
        float_columns = [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC',
            'Overall Score',
            'Vectorization Score',
            'Fathomnet Score',
            'Feedback Score',
            'House Price Score',
            'Spaceship Titanic Score',
            'AMP Parkinsons Disease Progression Prediction Score',
            'CIFAR10 Score',
            'IMDB Score',
            'Level 1 Accuracy',
            'Level 2 Accuracy',
            'Level 3 Accuracy',
            'Total Cost'
        ]
        
        for column in float_columns:
            if column in results_df.columns:
                try:
                    results_df[column] = results_df[column].round(2)
                except Exception as e:
                    print(f"Error rounding {column}: {e}")
        
        return results_df

    def check_token_usage_data(self, benchmark_name):
        """Debug helper to check token usage data"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT * FROM token_usage
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df

    def get_models_for_benchmark(self, benchmark_name):
        """Get list of unique model names used in a benchmark"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT DISTINCT model_name
                FROM token_usage
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df['model_name'].tolist()

    def get_all_agents(self, benchmark_name):
        """Get list of all agent names for a benchmark"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT DISTINCT agent_name
                FROM parsed_results
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df['agent_name'].tolist()

    def get_total_benchmarks(self):
        """Get the total number of unique benchmarks in the database"""
        benchmarks = set()
        for db_file in self.db_dir.glob('*.db'):
            benchmarks.add(db_file.stem.replace('_', '/'))
        return len(benchmarks) - 2 # TODO hardcoded -1 for mlagentbench and agentharm benign

    def get_total_agents(self):
        """Get the total number of unique agents across all benchmarks"""
        total_agents = set()
        # Use the parsed_results table since it's guaranteed to have all benchmark-agent pairs
        for db_file in self.db_dir.glob('*.db'):
            # skip mlagentbench
            if db_file.stem == 'mlagentbench':
                continue # TODO remove hardcoded skip for mlagentbench
            benchmark_name = db_file.stem.replace('_', '/')
            with self.get_conn(benchmark_name) as conn:
                query = '''
                    SELECT DISTINCT benchmark_name, agent_name 
                    FROM parsed_results
                '''
                
                results = conn.execute(query).fetchall()
                # Add each benchmark-agent pair to the set
                total_agents.update(results)
        return len(total_agents)

    def get_agent_url(self, agent_name, benchmark_name):
        """Get the URL for an agent from the metadata file."""
        try:
            with open('agents_metadata.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
                if benchmark_name in metadata:
                    for agent in metadata[benchmark_name]:
                        if agent['agent_name'] == agent_name:
                            return agent.get('url', '')
        except Exception as e:
            print(f"Error getting agent URL: {e}")
        return ''

    def get_traces_url(self, agent_name, benchmark_name, run_id):
        """Get the download URL for agent traces."""
        # This would typically point to where traces are stored, e.g. HuggingFace
        return f"https://huggingface.co/datasets/princeton-nlp/hal-eval-results/resolve/main/traces/{benchmark_name}/{agent_name}/{run_id}.json"

if __name__ == '__main__':
    preprocessor = TracePreprocessor()
    preprocessor.preprocess_traces()