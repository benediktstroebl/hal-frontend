import os
import glob
from rich.table import Table
from operator import itemgetter
from openai import AsyncOpenAI, AsyncAzureOpenAI
from collections import defaultdict
# import weave
from pydantic import BaseModel
from abc import ABC, abstractmethod
import json
from typing import Dict, List
from datetime import datetime
import backoff
from openai import APITimeoutError, APIError, RateLimitError
from dotenv import load_dotenv
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich.logging import RichHandler
import asyncio
import tiktoken
import weave


# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
log = logging.getLogger("agent_monitor")

load_dotenv()


class FailureCategory(BaseModel):
    category_id: int
    category_name: str
    description: str

class FailureCategories(BaseModel):
    failure_categories: list[FailureCategory]

class TaskSummary(BaseModel):
    task_id: str
    summary: str

class TaskClassification(BaseModel):
    task_id: str
    category_id: str
    category_name: str
    explanation: str

class OverallAnalysis(BaseModel):
    failure_categories: List[Dict]
    task_classifications: Dict[str, Dict]
    summary: str

class AsyncLLMClient(ABC):
    @abstractmethod
    async def generate_text(self, prompt, system_message=None, response_format=None):
        pass


class AsyncOpenAIClient(AsyncLLMClient):
    def __init__(self, model="gpt-4o-mini", max_tries=5, max_time=300, max_concurrent=50):
        self.model = model
        self.client = AsyncOpenAI()
        self.max_tries = max_tries
        self.max_time = max_time
        self.semaphore = asyncio.Semaphore(max_concurrent)

    @backoff.on_exception(
        backoff.expo,
        (APITimeoutError, APIError, RateLimitError),
        max_tries=10,
        max_time=300
    )
    async def _make_request(self, messages, response_format=None):
        if response_format:
            return await self.client.beta.chat.completions.parse(
                model=self.model, 
                messages=messages, 
                response_format=response_format
            )
        else:
            return await self.client.chat.completions.create(
                model=self.model, 
                messages=messages
            )

    async def generate_text(self, prompt, system_message=None, response_format=None):
        async with self.semaphore:  # Control concurrent access
            messages = [
                {"role": "system", "content": system_message or "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
            try:
                response = await self._make_request(messages, response_format)
                return response.choices[0].message.content
            except Exception as e:
                raise Exception(f"Failed after {self.max_tries} attempts or {self.max_time} seconds: {str(e)}")

# def get_weave_calls(client):
#     calls = client.calls()
#     processed_calls = []
#     for call in calls:
#         ChatCompletion = weave.ref(call.output).get()
#         choices = [choice.message.content for choice in ChatCompletion.choices]
#         output = {
#             'weave_task_id': call.attributes['weave_task_id'],
#             'trace_id': call.trace_id,
#             'project_id': call.project_id,
#             'created_timestamp': ChatCompletion.created,
#             'inputs': dict(call.inputs),
#             'id': call.id,
#             'outputs': {'choices' : choices},
#             'exception': call.exception,
#             'summary': call.summary,
#             'display_name': call.display_name,
#             'attributes': dict(call.attributes),
#             "_children": call._children,
#             '_feedback': call._feedback,
#         }
#         processed_calls.append(output)
#     return processed_calls

async def analyze_agent_performance(processed_calls, failed_tasks: list, llm_client):
    log.info(f"Starting analysis of {len(failed_tasks)} failed tasks")
    task_calls = defaultdict(list)
    for call in processed_calls:
        if 'call_data' in call:
            call_2 = call['call_data']
            if 'weave_task_id' in call_2:
                if call_2['weave_task_id'] in failed_tasks:
                    task_calls[call_2['weave_task_id']].append(call_2)
        elif 'weave_task_id' in call:
            if call['weave_task_id'] in failed_tasks:
                task_calls[call['weave_task_id']].append(call)
        else:
            if call['id'] in failed_tasks:
                task_calls[call['id']].append(call)
    for task_id in task_calls:
        if 'created_timestamp' in task_calls[task_id][0]:
            task_calls[task_id].sort(key=lambda x: x['created_timestamp'])
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        # Create progress tasks
        summarize_task_progress = progress.add_task("Summarizing tasks...", total=len(task_calls))
        
        # Process tasks in parallel but with controlled concurrency
        tasks = []
        for task_id, calls in task_calls.items():
            tasks.append(summarize_task(task_id, calls, llm_client))
        
        task_summaries = await asyncio.gather(*tasks)
        progress.update(summarize_task_progress, completed=len(task_calls))
        
        log.info("Identifying failure categories...")
        failure_categories = await identify_failure_categories(task_summaries, llm_client)
        
        classify_progress = progress.add_task("Classifying tasks...", total=len(task_summaries))
        
        # Process classifications in parallel but with controlled concurrency
        classification_tasks = []
        for summary in task_summaries:
            classification_tasks.append(classify_task(summary, failure_categories, llm_client))
        
        task_classifications = await asyncio.gather(*classification_tasks)
        progress.update(classify_progress, completed=len(task_summaries))
        
        log.info("Generating overall summary...")
        overall_summary = await generate_overall_summary(failure_categories, task_classifications, llm_client)

    task_classifications = {tc["task_id"]: tc for tc in task_classifications}
    
    log.info("Analysis complete!")
    return dict(OverallAnalysis(
        failure_categories=failure_categories,
        task_classifications=task_classifications,
        summary=overall_summary
    ))

async def summarize_task(task_id, calls, llm_client):
    calls_summary = ""
    if 'created_timestamp' in calls[0]:
        for i, call in enumerate(calls, 1):
            calls_summary += f"""
            Step {i}:
            Input: {call['inputs']}
            Output: {call['outputs']}
            Timestamp: {datetime.fromtimestamp(call['created_timestamp'])}
            """
    else:
        for call in calls:
            calls_summary = f"""
            Number of steps: {len(call['messages'])}"""
            for message in call['messages']:
                calls_summary += f"""
                Step: {message}
                """
            calls_summary += f"""
            Output: {call['output']['choices'][0]['message']}
            """
            
            break
        
        
    # Check token length of calls_summary
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(calls_summary)
    if len(tokens) > 110000:
        # cut from the beginning until it fits
        while len(encoding.encode(calls_summary)) > 110000:
            calls_summary = calls_summary.split('\n')[1:]
            calls_summary = '\n'.join(calls_summary)

    prompt = f"""
    Summarize the AI agent's performance on the following task:
    Task ID: {task_id}
    Number of steps: {len(calls)}

    Detailed steps:
    {calls_summary}

    Provide a brief summary of:
    1. The main goal of the task (inferred from the inputs and outputs)
    2. The agent's approach, including key steps and decisions made
    3. Any significant challenges or errors encountered during the task
    4. The final outcome why the task failed. Be detailed about the reason for failure.

    Keep the summary concise (around 200 words) but include specific details about the agent's performance and any notable aspects of its problem-solving process.
    """

    system_message = "You are an AI performance analyst tasked with summarizing an AI agent's performance on individual tasks. Focus on the most important aspects of the agent's approach and performance."
    summary = await llm_client.generate_text(prompt, system_message, response_format=TaskSummary)
    return json.loads(summary)

async def identify_failure_categories(task_summaries, llm_client):
    summaries_text = "\n\n".join([f"Task {s['task_id']}:\n{s['summary']}" for s in task_summaries])
    prompt = f"""
    Analyze the following summaries of an AI agent's performance across multiple tasks:

    {summaries_text}

    Identify recurring categories of failures that the agent faces across these tasks. For each category:
    1. Provide a short, descriptive name (max 5 words)
    2. Write a brief description explaining the nature of this failure or challenge category

    Focus on patterns that appear across multiple tasks and represent specific errors that impacted the agent's performance. Make sure that your categories are distinct and cover a range of recurring issues. The categories should not bee too general.

    Examples for categories could include:
    Incorrect Implementation - The agent made a change to a reasonable area but their solution didn’t correctly address the issue.
    Gave Up Prematurely - The agent decides to stop solving the task after encountering some difficulty.
    Failed Edit Recovery - The agent went into an loop, making recurrent failing edits without recovering.
    """

    system_message = "You are an expert in AI agent analysis, tasked with identifying recurring patterns in agent performance across multiple tasks."
    categories = await llm_client.generate_text(prompt, system_message, response_format=FailureCategories)
    return [dict(category) for category in json.loads(categories)['failure_categories']]

# Split classify_tasks into individual task classification for better progress tracking
async def classify_task(task_summary, failure_categories, llm_client):
    categories_text = "\n".join([f"{cat['category_id']}. {cat['category_name']}: {cat['description']}" 
                               for i, cat in enumerate(failure_categories)])
    
    prompt = f"""
    Failure Categories:
    {categories_text}

    Task Summary:
    {task_summary['summary']}

    Classify this task into one of the failure categories listed above. Provide:
    1. The number of the chosen category
    2. A brief explanation of why this category best fits the task's outcome

    If the task doesn't clearly fit any category, you may classify it as "0. Other" and explain why.
    """

    system_message = "You are an AI performance analyst tasked with classifying task outcomes into predefined categories."
    classification = await llm_client.generate_text(prompt, system_message, response_format=TaskClassification)
    classification = json.loads(classification)
    
    category_number = classification['category_id']
    category_name = "Other" if str(category_number) == "0" else next(
        (cat['category_name'] for cat in failure_categories 
         if str(cat['category_id']) == str(category_number)), 
        "Other"
    )

    return dict(TaskClassification(
        task_id=task_summary['task_id'],
        category_id=category_number,
        category_name=category_name,
        explanation=classification['explanation']
    ))

async def generate_overall_summary(failure_categories, task_classifications, llm_client):
    categories_text = "\n".join([f"{cat['category_name']}: {cat['description']}" for cat in failure_categories])

    classifications_text = "\n".join([f"Task {tc['task_id']}: {tc['category_name']}" for tc in task_classifications])

    prompt = f"""
    Failure Categories:
    {categories_text}

    Task Classifications:
    {classifications_text}

    Based on the failure categories identified and the classification of tasks, provide an overall summary of the AI agent's performance across all tasks. Include:
    1. The most common types of failures or challenges
    2. Any patterns in the agent's performance across different tasks
    3. Suggestions for areas of improvement in the agent's design or training

    Keep the summary concise but insightful, focusing on the most significant findings and their implications for AI agent development. Do only return the summary itself without any preceding context etc.
    """

    system_message = "You are a senior AI researcher tasked with providing a high-level analysis of an AI agent's performance across multiple tasks."
    return await llm_client.generate_text(prompt, system_message)

class BenchmarkRun:
    def __init__(self, file_path):
        self.file_path = file_path
        # Only load metadata initially
        with open(file_path) as f:
            data = json.load(f, object_hook=self._metadata_only)
        self.config = data.get('config', {})
        self.agent_name = self.config.get('agent_name', 'unknown')
        self.run_id = self.config.get('run_id', 'unknown')
        self.benchmark = self.config.get('benchmark_name', 'unknown')
        self.accuracy = data.get('results', {}).get('accuracy', 0)
        self._data = None  # Full data loaded on demand
    
    def _metadata_only(self, obj):
        """Custom object hook to load only metadata"""
        if 'config' in obj:
            return {
                'config': obj['config'],
                'results': {'accuracy': obj.get('results', {}).get('accuracy', 0)}
            }
        return obj
    
    @property
    def data(self):
        """Lazy load full data only when needed"""
        if self._data is None:
            log.info(f"Loading full data for {self.file_path}")
            with open(self.file_path) as f:
                self._data = json.load(f)
        return self._data

def get_top_runs_per_benchmark():
    # Get all json files in evals_live
    json_files = glob.glob('evals_live/*.json')
    log.info(f"Found {len(json_files)} evaluation files")
    
    # Group runs by benchmark
    benchmark_runs = defaultdict(list)
    for file_path in json_files:
        try:
            run = BenchmarkRun(file_path)
            benchmark_runs[run.benchmark].append(run)
        except json.JSONDecodeError:
            log.warning(f"Failed to parse {file_path}")
            continue
        except Exception as e:
            log.warning(f"Error processing {file_path}: {str(e)}")
            continue
    
    # For each benchmark, get top 2 unique agents
    selected_runs = {}
    for benchmark, runs in benchmark_runs.items():
        try:
            log.info(f"Processing benchmark: {benchmark} ({len(runs)} runs)")
            # Sort runs by accuracy
            sorted_runs = sorted(runs, key=lambda x: x.accuracy, reverse=True)
            
            # Get top 2 runs with different agent names
            unique_agents = []
            seen_agents = set()
            for run in sorted_runs:
                if run.agent_name not in seen_agents:
                    unique_agents.append(run)
                    seen_agents.add(run.agent_name)
                    if len(unique_agents) == 2:
                        break
            
            if len(unique_agents) == 2:
                selected_runs[benchmark] = unique_agents
                log.info(f"Selected agents for {benchmark}: {[run.agent_name for run in unique_agents]}")
            else:
                log.warning(f"Could not find 2 unique agents for {benchmark}")
        except Exception as e:
            log.warning(f"Error processing {benchmark}: {str(e)}")
            continue
    
    return selected_runs

async def process_benchmarks():
    # Get top runs for each benchmark
    selected_runs = get_top_runs_per_benchmark()
    
    # Display table of selected runs
    table = Table(title="Selected Runs for Analysis")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Agent", style="green")
    table.add_column("Run ID", style="yellow")
    table.add_column("Accuracy", style="magenta")
    
    for benchmark, runs in selected_runs.items():
        for run in runs:
            table.add_row(
                benchmark,
                run.agent_name,
                run.run_id,
                f"{run.accuracy:.2%}"
            )
    
    console.print(table)
            
    # Process each run
    openai_client = AsyncOpenAIClient(max_concurrent=5)
    
    for benchmark, runs in selected_runs.items():
        for run in runs:
            log.info(f"Processing {run.agent_name} on {benchmark}")
            
            if isinstance(run.data.get('failure_report', None), dict):
                log.info(f"Failure report already exists for {run.agent_name} on {benchmark}")
                continue
            
            # if run.data['raw_logging_results'] is dict, then we need to load the file
            processed_calls = []
            if isinstance(run.data['raw_logging_results'], dict):
                # append all steps from all keys
                for key in run.data['raw_logging_results'].keys():
                    processed_calls.extend(run.data['raw_logging_results'][key]['steps'])
            elif isinstance(run.data['raw_logging_results'], str):
                processed_calls = run.data['raw_eval_results']['samples']
            else:
                processed_calls = run.data['raw_logging_results']
            
            overall_analysis = await analyze_agent_performance(
                processed_calls, 
                failed_tasks=run.data['results']['failed_tasks'],
                llm_client=openai_client
            )
            
            # Add failure report to the original data
            run.data['failure_report'] = overall_analysis
            
            # Save updated json
            log.info(f"Saving updated results to {run.file_path}")
            with open(run.file_path, 'w') as f:
                json.dump(run.data, f, indent=4)
            
            # Print summary
            console.print(f"\nFailure Analysis Summary for {run.agent_name} on {benchmark}:")
            console.print("=" * 80)
            console.print(overall_analysis['summary'], style="bold green")
            console.print("=" * 80)

async def main():
    log.info("Starting benchmark analysis")
    await process_benchmarks()
    log.info("Analysis complete!")

if __name__ == "__main__":
    weave.init("failure_report")
    asyncio.run(main())